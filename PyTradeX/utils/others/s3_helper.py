import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from io import BytesIO
import s3fs
import boto3
from botocore.exceptions import ClientError
import pickle
import json
import yaml
from datetime import datetime, timezone
from tqdm import tqdm
import logging
import shutil
import zipfile
import os
from pathlib import Path
from tqdm import tqdm
from typing import Set, List, Dict, Tuple
from pprint import pprint, pformat


def get_secrets(secret_name: str = 'access_keys') -> str:
    # Create a Secrets Manager client
    session = boto3.session.Session()
    secrets_client = session.client(
        service_name='secretsmanager',
        region_name=REGION
    )

    try:
        get_secret_value_response = secrets_client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    secret = json.loads(get_secret_value_response['SecretString'])

    return secret

# Define region
REGION = "sa-east-1"

# Extract secrets
ACCESS_KEYS = get_secrets(secret_name='access_keys')

# Create an S3 client instance
# print('Instanciating S3_CLIENT.\n')
S3_CLIENT = boto3.client(
    's3',
    region_name=REGION,
    aws_access_key_id=ACCESS_KEYS["AWS_ACCESS_KEY_ID"], # os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=ACCESS_KEYS["AWS_SECRET_ACCESS_KEY"], # os.environ.get("AWS_SECRET_ACCESS_KEY")
)

# Create an s3fs.S3FileSystem instance
FS = s3fs.S3FileSystem(
    key=ACCESS_KEYS["AWS_ACCESS_KEY_ID"], # os.environ.get("AWS_ACCESS_KEY_ID"),
    secret=ACCESS_KEYS["AWS_SECRET_ACCESS_KEY"], # os.environ.get("AWS_SECRET_ACCESS_KEY"),
    anon=False  # Set to True if your bucket is public
)

# Load default partition & intervals
with open(os.path.join("config", "config.yaml")) as file:
    config: Dict[str, Dict[str, str]] = yaml.load(file, Loader=yaml.FullLoader)

DEFAULT_PARTITION = config.get('data_params').get('partition_cols')
INTERVALS = config.get('general_params').get('intervals')


def find_filters(
    load_reduced_dataset: bool = False,
    partition_cols: List[str] = DEFAULT_PARTITION,
    added_filters: List[Tuple[str, str, List[str]]] = None
) -> List[Tuple[str, str, List[str]]]:
    """
    Function designed to find the least amount of partitions required to load (at least) 1.5
    months of observations.
    """
    # Define dafault filters
    filters = None

    if partition_cols is None:
        return filters
    
    if load_reduced_dataset:
        if (
            'year_quarter' in partition_cols 
            or 'year_bimester' in partition_cols
            or 'year_month' in partition_cols
        ):
            # Find current year, quarter, bimester, month & day
            current_year = datetime.today().year
            current_quarter = int(np.ceil(datetime.today().month / 3))
            current_bimester = int(np.ceil(datetime.today().month / 2))
            current_month = datetime.today().month
            current_day = datetime.today().day

            if partition_cols == ['year_quarter']:
                # Find current year_quarter
                year_quarters = [f'{current_year}_{current_quarter}']

                if current_month == 1:
                    # If it's January, include the last quarter from the previous year
                    year_quarters.append(f'{current_year-1}_4')

                elif current_month == 2:
                    # If it's February, include last quarter from last year, only if less than two
                    # weeks have passed
                    if current_day < 14:
                        year_quarters.append(f'{current_year-1}_4')

                elif current_month in [4, 7, 10]:
                    # If it's April, July or October include this year's last quarter
                    year_quarters.append(f'{current_year}_{current_quarter-1}')

                elif current_month in [5, 8, 11]:
                    # If it's May, August or November, include this year's last quarter, only if less
                    # than two weeks have passed
                    if current_day < 14:
                        year_quarters.append(f'{current_year}_{current_quarter-1}')

                # Define year_quarter filters to apply
                filters = [('year_quarter', 'in', year_quarters)]

            elif partition_cols == ['year_bimester']:
                # Find current year_bimester
                year_bimesters = [f'{current_year}_{current_bimester}']

                if current_month == 1:
                    # If it's January, then include the last bimester from the previous year
                    year_bimesters.append(f'{current_year-1}_6')

                elif current_month == 2:
                    # If it's February, include last bimester from last year, only if less than two
                    # weeks have passed
                    if current_day < 14:
                        year_bimesters.append(f'{current_year-1}_6')

                elif current_month in [3, 5, 7, 9, 11]:
                    # If it's March, May, July, September or November include this year's last bimester
                    year_bimesters.append(f'{current_year}_{current_bimester-1}')
                
                elif current_month in [4, 6, 8, 10, 12]:
                    # If it's April, June, August, October or December include this year's last bimester,
                    # only if less than two weeks have passed
                    if current_day < 14:
                        year_bimesters.append(f'{current_year}_{current_bimester-1}')

                # Define year_bimester filters to apply
                filters = [('year_bimester', 'in', year_bimesters)]

            elif partition_cols == ['year_month']:
                # Find current & last year_months
                year_months = [f'{current_year}_{current_month}']

                if current_month == 1:
                    # If it's January, include December from last year
                    year_months.append(f'{current_year-1}_12')
                else:
                    # If it's not January, include this year's previous month
                    year_months.append(f'{current_year}_{current_month-1}')

                # Until this point, there will be at least 1 month (and one day) of observations and at most,
                # two complete months.
                if current_day < 14:
                    # If less than 14 days have passed, I need to include more observations to have more than
                    # 1.5 observations of data
                    if current_month == 1:
                        # Include last year's November partition
                        year_months.append(f'{current_year-1}_11')
                    elif current_month == 2:
                        # Include last year's December partition
                        year_months.append(f'{current_year-1}_12')
                    else:
                        # Include partition from two months ago
                        year_months.append(f'{current_year}_{current_month-2}')

                # Define year_month filters to apply
                filters = [('year_month', 'in', year_months)]
                
        if added_filters is not None:
            if filters is None:
                filters = added_filters
            else:
                filters.extend(added_filters)

    return filters


def load_from_s3(
    path: str,
    load_reduced_dataset: bool = False,
    partition_cols: List[str] = DEFAULT_PARTITION,
    added_filters: List[Tuple[str, str, List[str]]] = None,
    ignore_checks: bool = False
):
    # Extract bucket, key & format
    bucket, key = path.split('/')[0], '/'.join(path.split('/')[1:])
    read_format = key.split('.')[-1]

    if read_format == 'parquet':
        # Remove extention
        prefix = key.replace(".parquet", "")

        # Find filters
        filters = find_filters(
            load_reduced_dataset=load_reduced_dataset,
            partition_cols=partition_cols,
            added_filters=added_filters
        )        

        # Find files
        files = FS.glob(f's3://{bucket}/{prefix}/*/*.parquet')
        if len(files) == 0:
            files = FS.glob(f's3://{bucket}/{prefix}/*/*/*.parquet')
            if len(files) == 0:
                files = f"s3://{bucket}/{prefix}/dataset-0.parquet"

        # Create a Parquet dataset
        dataset = pq.ParquetDataset(
            path_or_paths=files,
            filesystem=FS,
            filters=filters
        )
        
        # Read the dataset into a Pandas DataFrame
        asset: pd.DataFrame = dataset.read_pandas().to_pandas()

        # Sort index, drop duplicated indexes & drop unrequired columns
        drop_cols = [
            'month',
            'bimester',
            'quarter',
            'year',
            'year_month',
            'year_bimester',
            'year_quarter'
        ]

        asset: pd.DataFrame = (
            asset
            .sort_index(ascending=True)
            .loc[~asset.index.duplicated(keep='last')]
            .drop(columns=drop_cols, errors='ignore')
        )

        if not ignore_checks:
            if isinstance(asset.index, pd.DatetimeIndex):
                # Assert there are no missing rows
                # freq = {
                #     '30min': '30min',
                #     '60min': '60min',
                #     '1d': '1D'
                # }[INTERVALS]

                # full_idx = pd.date_range(asset.index.min(), asset.index.max(), freq=freq)

                # assert len(list(set(full_idx) - set(asset.index))) == 0, f"Missing index: \n{pformat(set(full_idx) - set(asset.index))}\n\n"

                # Assert there are at least 40 days of observations
                days = (asset.index.max() - asset.index.min()).days
                assert days >= 40, f"Days between max & min indexes: {days} - {path.split('/')[-1]}"

            # Assert no unwanted columns were loaded
            assert len([c for c in drop_cols if c in asset.columns]) == 0

    elif read_format == 'pickle':
        # Retrieve stored object
        obj: dict = S3_CLIENT.get_object(
            Bucket=bucket,
            Key=key
        )

        # Read pickle
        asset: dict = pickle.loads(
            BytesIO(obj['Body'].read()).read()
        )
    elif read_format == 'json':
        # Retrieve stored object
        obj: dict = S3_CLIENT.get_object(
            Bucket=bucket,
            Key=key
        )

        # Read json
        asset: dict = json.loads(
            BytesIO(obj['Body'].read()).read()
        )
    else:
        raise Exception(f'Invalid "read_format" parameter: {read_format}, extracted from path: {path}.\n\n')
    
    assert len(asset) > 0, f"Loaded asset from s3://{bucket}/{prefix} contains zero keys. {asset}"

    return asset


def write_to_s3(
    asset, 
    path: str,
    partition_cols: List[str] = DEFAULT_PARTITION,
    datetime_col: str = None,
    overwrite: bool = True
):
    # Extract bucket, key & format
    bucket, key = path.split('/')[0], '/'.join(path.split('/')[1:])
    write_format = key.split('.')[-1]

    if write_format == 'parquet':
        asset: pd.DataFrame = asset.copy(deep=True)
        prefix = key.replace(".parquet", "")

        # Extract existing_data_behavior
        if overwrite:
            # Delete all found files before writing a new one
            existing_data_behavior = 'delete_matching'
        else:
            # (Append) Overwrite new partitions while leaving old ones
            existing_data_behavior = 'overwrite_or_ignore'

        # Write PyArrow Table as a parquet file, partitioned by year_quarter
        # else:
        if partition_cols is not None:
            # Define datetime_series
            if datetime_col is None:
                datetime_series = asset.index.to_series()
            else:
                datetime_series = asset[datetime_col].copy()

            if asset.shape[0] > 0:
                if 'year_month' in partition_cols:
                    # Write month, year & year_month columns
                    asset['month'] = datetime_series.dt.month.astype(str)
                    asset['year'] = datetime_series.dt.year.astype(str)

                    asset['year_month'] = asset['year'] + "_" + asset['month']

                if 'year_bimester' in partition_cols:
                    # Write bimester, year & year_bimester columns
                    asset['bimester'] = np.ceil(datetime_series.dt.month / 2).astype(int).astype(str)
                    asset['year'] = datetime_series.dt.year.astype(str)

                    asset['year_bimester'] = asset['year'] + "_" + asset['bimester']

                if 'year_quarter' in partition_cols:
                    # Write quarter, year & year_quarter columns
                    asset['quarter'] = datetime_series.dt.quarter.astype(str)
                    asset['year'] = datetime_series.dt.year.astype(str)

                    asset['year_quarter'] = asset['year'] + "_" + asset['quarter']
            else:
                if 'year_month' in partition_cols:
                    # Write dummy year_month column
                    asset['year_month'] = None

                if 'year_bimester' in partition_cols:
                    # Write dummy year_bimester column
                    asset['year_bimester'] = None

                if 'year_quarter' in partition_cols:
                    # Write dummy year_quarter column
                    asset['year_quarter'] = None

        if overwrite:
            if partition_cols is None:
                delete_from_s3(path=f'{bucket}/{prefix}/dataset-0.parquet')
            else:
                # Delete objects
                delete_s3_directory(
                    bucket=bucket, 
                    directory=prefix
                )
        
        pq.write_to_dataset(
            pa.Table.from_pandas(asset),
            root_path=f's3://{bucket}/{prefix}',
            partition_cols=partition_cols,
            filesystem=FS,
            schema=pa.Schema.from_pandas(asset),
            basename_template='dataset-{i}.parquet',
            use_threads=True,
            compression='snappy',
            existing_data_behavior=existing_data_behavior
        )
    elif write_format == 'pickle':
        # Delete object
        delete_from_s3(path=path)

        # Save new object
        S3_CLIENT.put_object(
            Bucket=bucket,
            Key=key,
            Body=pickle.dumps(asset)
        )
    elif write_format == 'json':
        # Delete object
        delete_from_s3(path=path)

        # Save new object
        S3_CLIENT.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(asset)
        )
    else:
        raise Exception(f'Invalid "write_format" parameter: {write_format}, extracted from path: {path}.\n\n')
    

def delete_from_s3(
    path: str
):
    bucket, key = path.split('/')[0], '/'.join(path.split('/')[1:])

    S3_CLIENT.delete_object(
        Bucket=bucket, 
        Key=key
    )


def find_keys(
    bucket: str,
    subdir: str = None,
    include_additional_info: bool = False
) -> Set[str]:
    # Validate subdir
    if subdir is None:
        subdir = ''

    # Define keys to populate
    s3_keys = set()

    # Find dirs
    prefixes = S3_CLIENT.list_objects_v2(
        Bucket=bucket,
        Prefix=subdir, 
        Delimiter='/'
    ).get('CommonPrefixes')

    if prefixes is not None:
        prefixes = [p['Prefix'] for p in prefixes]

        for prefix in prefixes:
            # Find prefix contents
            contents = S3_CLIENT.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix
            ).get('Contents', [])
        
            if len(contents) > 0:
                if not include_additional_info:
                    s3_keys = s3_keys | {
                        content['Key'] for content in contents
                        if not(content['Key'].endswith('/'))
                    }
                else:
                    s3_keys = s3_keys | {
                        (content['Key'], content['Size'], content['LastModified']) for content in contents
                        if not(content['Key'].endswith('/'))
                    }

        # print('s3_keys:')
        # pprint(s3_keys)
        # print('\n\n')
        
        return s3_keys
    print(f'[WARNING] No keys were found for bucket: {bucket}, subdir: {subdir}.\n')
    return {}


def find_prefixes(
    bucket: str, 
    prefix: str = None, 
    results: set = set(),
    debug: bool = False
):
    if prefix is None:
        prefix = ''

    if debug:
        print(f'bucket: {bucket}\n'
              f'prefix: {prefix}\n\n')

    result: dict = S3_CLIENT.list_objects_v2(
        Bucket=bucket, 
        Prefix=prefix, 
        Delimiter='/'
    )
    if debug:
        print(f'result')
        pprint(result)
        print('\n\n')
    
    for common_prefix in result.get('CommonPrefixes', []):
        subdir = common_prefix.get('Prefix')
        results.add(subdir)
        
        # Recursively list subdirectories
        find_prefixes(bucket, subdir, results)

    return results


def delete_s3_directory(
    bucket, 
    directory
):
    # List objects with the common prefix
    objects = S3_CLIENT.list_objects_v2(Bucket=bucket, Prefix=directory)
    
    # Check if there are objects to delete
    if 'Contents' in objects:
        for obj in objects['Contents']:
            # print(f'deleating: {obj["Key"]}')
            S3_CLIENT.delete_object(Bucket=bucket, Key=obj['Key'])

    # Check if there are subdirectories (common prefixes) to delete
    if 'CommonPrefixes' in objects:
        for subdir in objects['CommonPrefixes']:
            delete_s3_directory(bucket, subdir['Prefix'])

    # Finally, delete the common prefix (the "directory" itself)
    S3_CLIENT.delete_object(Bucket=bucket, Key=directory)

    # print('\n')


def sincronize_buckets(
    source_bucket: str, 
    destination_bucket: str, 
    sub_dir: str = None,
    debug: bool = False
):
    """
    Objects
    """
    # Find destination objects
    dest_objects = find_keys(
        bucket=destination_bucket,
        include_additional_info=False
    )
    while len(dest_objects) > 0:
        if debug:
            print(f'dest_objects:')
            pprint(dest_objects)
            print('\n\n')

        # Remove destination objects
        print(f'Removing objects from {destination_bucket}:')
        for obj in tqdm(dest_objects):
            # print(f"Removing: {destination_bucket}/{obj}")
            delete_from_s3(path=f"{destination_bucket}/{obj}")

        # Re-setting dest_objects
        dest_objects = find_keys(
            bucket=destination_bucket,
            include_additional_info=False
        )
    
    print(f'\nAll objects in {destination_bucket} have been removed.\n\n')
    
    # Find source objects
    source_objects = find_keys(
        bucket=source_bucket,
        include_additional_info=False
    )
    
    while len(source_objects - dest_objects) > 0:
        # Copy source objects to destination bucket
        print(f"Copying: objects from {source_bucket} to {destination_bucket}")
        for obj in tqdm(source_objects - dest_objects):
            # print(f"Copying: {obj} from {source_bucket} to {destination_bucket}")
            S3_CLIENT.copy_object(
                Bucket=destination_bucket, 
                CopySource={
                    'Bucket': source_bucket, 
                    'Key': obj
                },
                Key=obj
            )
        
        # Re-setting dest_objects
        dest_objects = find_keys(
            bucket=destination_bucket,
            include_additional_info=False
        )
    
    print(f'\n{destination_bucket} has been filled.\n\n')


def update_pytradex_zip(
    bucket_name: str,
    logger: logging.Logger
):
    def remove_pytradex_zip():
        try:
            os.remove('pytradex_lambda.zip')
        except Exception as e:
            logger.warning('Unable to remove pytradex_lambda.zip: %s', e)

    def find_base_repo_root() -> Path:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if 'PyTradeX' in base_path:
            base_path = base_path[:base_path.find('PyTradeX')]
        
        base_path = Path(base_path)

        return base_path / "PyTradeX"

    def zip_pytradex(cwd: Path):
        def ignore_root(r):
            ignore_list = [
                '.ipynb_checkpoints', 
                'bin', 'etc', 'include', 'share',
                '.streamlit',
                '.vscode',
                'docker', 
                'docs', 
                'dummy_bucket', 
                'resources', 
                'test', 
                'workflows', 
                '.git', 
                '__pycache__'
            ]
            if any([ig in r for ig in ignore_list]):
                return True
            return False

        with zipfile.ZipFile('pytradex_lambda.zip', 'w', zipfile.ZIP_DEFLATED) as zipf: # filename = 'pytradex_lambda'
            for root, dirs, files in tqdm(os.walk(cwd)): # dirname = cwd
                if not ignore_root(root):
                    for file in files:
                        if file != 'pytradex_lambda.zip':
                            zipf.write(
                                os.path.join(root, file),
                                os.path.relpath(os.path.join(root, file), os.path.join(cwd, '..')) # dirname = cwd
                            )
    
    def upload_zip_pytradex():
        # Read the file contents
        with open("pytradex_lambda.zip", 'rb') as f:
            file_content = f.read()
        
        # Upload zip file to S3
        S3_CLIENT.put_object(
            Bucket=bucket_name,
            Key='lambda/pytradex_lambda.zip',
            Body=file_content
        )

        logger.info('pytradex_lambda.zip was added to %s.', bucket_name)
    
    # Remove old pytradex_lambda.zip
    remove_pytradex_zip()

    # Create new pytradex.zip file
    cwd: Path = find_base_repo_root()
    zip_pytradex(cwd=cwd)

    # Upload zip file to S3
    upload_zip_pytradex()


# shutil.make_archive(
#     base_name='pytradex', 
#     format='zip', 
#     root_dir=os.path.dirname('PyTradeX'),
#     # base_dir=os.path.basename(source.strip(os.sep)),
#     logger=logger
# )
