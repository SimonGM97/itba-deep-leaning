import os

def setup_keys():
    # Read Keys.txt
    # keys_path = 'config/Keys.txt'
    # with open(keys_path, 'r', encoding='utf-8', errors='ignore') as file:
    #     # Read the entire content of the file
    #     content = file.read()

    #     # Iterate over stored key-value pairs
    #     for pair in content.split('export'):
    #         if '=' in pair:
    #             key, val = pair.split('=')
    #             print(f'Setting {key}')
    #             os.environ[key] = val
    
    # ECR
    os.environ['ECR_REPOSITORY_NAME'] = 'pytradex-ecr'
    os.environ['ECR_REPOSITORY_URI'] = '097866913509.dkr.ecr.sa-east-1.amazonaws.com'
    os.environ['REGION'] = 'sa-east-1'

    # CodeCommit
    os.environ['CODE_COMMIT_PASSPHRASE'] = 'arrelocoo'

    # AWS Access
    os.environ['AWS_ACCESS_KEY_ID'] = 'AKIARNSKSS3SQZ3K7QW2'
    os.environ['AWS_SECRET_ACCESS_KEY'] = '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'

    # BINANCE
    os.environ['BINANCE_API_KEY'] = 'J9djw6bX5fNARUUeDpEczEaas0t7LrR7QbDz2eg27jIwpcEw2likqV4PzET8EIAS'
    os.environ['BINANCE_API_SECRET'] = 'lBdAiqaPf93BPmQf3HOelxSFoqQ2tn8jjD90wbfxNfi4OeTENUULidh4tQiQVEun'

    # LunarCrush
    os.environ['LUNAR_CRUSH_API_KEY'] = 'imhsiympkj8e55rufld8oq1ecl9yupntkk4m3ep'

    # KXY
    os.environ['KXY_API_KEY'] = 'pXdUYRKRYO9T0tLQ6jrU73oUZuwwMlOljD8Ob0T4'

