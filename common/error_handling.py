from enum import Enum, unique

__all__ = ['ResponseStatus', 'ServiceError']


@unique
class ResponseStatus(Enum):
    SUCCESS = (0, 'Service SUCCESS')
    ERROR_SERVICE_AVAILABLE = (-1, 'service temporarily unavailable')
    ERROR_REQUEST_NOT_JSON = (-1000, 'request body should be json format')
    ERROR_REQUEST_JSON_PARSE = (-1001, 'request json parse error')
    ERROR_MISSING_ARGS = (-1002, 'missing required arguments')
    ERROR_IVALID_ARG_VAL = (-1003, 'invalid argument value')
    ERROR_ARGUMENT_FORMAT = (-1004, 'argument format error')
    ERROR_INPUT_IMAGE_EMPTY = (-1100, 'input image is empty')
    ERROR_INPUT_IMAGE_BASE64 = (-1101, 'input image base64 error')
    ERROR_INPUT_IMAGE_READ = (-1102, 'input image read error')
    ERROR_INPUT_IMAGE_CHECKSUM = (-1103, 'input image checksum error')
    ERROR_INPUT_IMAGE = (-1104, 'input image error')
    ERROR_INPUT_IMAGE_HEADER = (-1105, 'input image header error')
    ERROR_INPUT_IMAGE_SIZE = (-1106, 'input image size is too large')
    ERROR_INPUT_IMAGE_CN = (-1107, 'input image channel number error, only support 1,3,4')
    ERROR_INPUT_IMAGE_FORMAT =(-1108, 'input image format error, only support "jpg,jpeg,png" format')
    ERROR_PREDICT = (-1200, 'predict error')
    ERROR_BATCH_PREDICT = (-1201, 'batch predict error')
    ERROR_UNKNOWN = (9999, 'unknown error')

    @property
    def code(self):
        return self.value[0]

    @property
    def message(self, msg=""):
        return self.value[1] + (", " + msg if msg else "")


class ServiceError(Exception):
    def __init__(self, error: ResponseStatus, extra_info: str=None):
        self.name = error.name
        self.code = error.code
        if extra_info is None:
            self.message = error.message
        else:
            self.message = f'{error.message}: {extra_info}'
        Exception.__init__(self)

    def __repr__(self):
        return f'[{self.__class__.__name__} {self.code}] {self.message}'

    __str__ = __repr__
