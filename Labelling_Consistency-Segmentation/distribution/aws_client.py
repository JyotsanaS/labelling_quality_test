import boto3
from distribution.download import S3Downloader
import distribution.aws_constants as aws_constants


class Aws:
    def __init__(self, role_arn=None, auth=None):
        self.role_arn = role_arn
        self.auth = auth
        self.s3_client = None
        self.session = None
        self.sts_client = None

    def get_session(self):
        self.session = (
            boto3.Session()
            if self.auth is None
            else boto3.Session(
                aws_access_key_id=self.auth[aws_constants.AWS_ACCESS_KEY_ID],
                aws_secret_access_key=self.auth[aws_constants.AWS_ACCESS_SECRET_KEY],
            )
        )
        return self.session

    def get_sts_client(self):
        session = self.get_session()
        self.sts_client = session.client(aws_constants.STS)
        return self.sts_client

    def get_s3_client(self):
        if self.role_arn is None:
            self.s3_client = self.get_session().client(aws_constants.S3)
        else:
            response = self.get_sts_client().assume_role(
                RoleArn=self.role_arn,
                RoleSessionName=aws_constants.S3_ACCESS_SESSION,
                DurationSeconds=3600,
            )
            # Extract the temporary credentials
            credentials = response[aws_constants.CREDENTIALS]
            # Create an S3 client using the temporary credentials

            self.s3_client = self.get_session().client(
                aws_constants.S3,
                aws_access_key_id=credentials[aws_constants.ACCESS_KEY_ID],
                aws_secret_access_key=credentials[aws_constants.SECRET_ACCESS_KEY],
                aws_session_token=credentials[aws_constants.SESSION_TOKEN],
            )
        return S3Downloader(self.s3_client)
