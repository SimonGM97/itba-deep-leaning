<div align="center">
<img src="./resources/logos/idea_4/logo_1_cut.jpeg" width="350">
</div>

&nbsp;
&nbsp;
# PyTradeX
PyTradeX is a propietary crypto currency forecasting library that is leveraged to run a trading bot that operates on Binance, trading cryptocurrency futures while following an ML based trading strategy.

&nbsp;
# Table of Contents

- [Installation](#installation)
- [Usage](#usage)

&nbsp;
# Installation

1. Install the AWS CLI v2 (if it's not already installed)
```bash
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg ./AWSCLIV2.pkg -target /
```
2. Set up the IAM credentials using aws configure:
```bash
aws configure
```
```
AWS Access Key ID: AWS_ACCESS_KEY_ID
AWS Secret Access Key: AWS_SECRET_ACCESS_KEY
Default region name: sa-east-1
Default output format: json
```
3. Clone the `PyTradeX` CodeCommit repository:
```bash
git clone https://github.com/SimonGM97/itba-deep-leaning.git
```
4. Create & activate python virtual environment:
```bash
python -m venv .itba_dl
source .itba_dl/bin/activate
```
5. Install the PyTradeX module in "editable" mode:
```bash
pip install -e .
```
  - *Note that this command will also install the dependencies, specified in `requirements.txt`.*
6. Install dask
```bash
pip install dask[dataframe]
```