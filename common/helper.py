import cv2
import base64
import numpy as np

from typing import Union, Optional
from dataclasses import dataclass
# from pydantic import BaseModel

from .error_handling import ResponseStatus, ServiceError

__all__ = ['RequestParser', 'ServerResponse']


def to_bgr(image):
    
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
    elif image.ndim == 3:
        num_channels = image.shape[-1]
        
        if num_channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif num_channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image


@dataclass
class RequestParser:
    """转换为算法的输入参数"""
    
    uid: Optional[str] = None  # request id 
    data: Union[str, bytes] = None # 图像文件数据, base64 编码
    fmt: Optional[str] = None # 数据格式
    param: Optional[str] = None # 输入参数

    @classmethod
    def convert2image(cls, data: Union[str, bytes, None]) -> Optional[np.ndarray]:

        try:
            if isinstance(data, str):
                data = data.encode()
                
            filename = base64.b64decode(data)
            image = cv2.imdecode(np.frombuffer(filename, dtype=np.uint8), -1)
            image = to_bgr(image)
        except Exception:
            raise ServiceError(ResponseStatus.ERROR_INPUT_IMAGE_READ)

        return image

    @classmethod    
    def parse_request(cls, requests):
        uid = requests.get('uid', None)
        data = requests.get('data', None)
        fmt = requests.get('fmt', None)
        param = requests.get('param', None)
        
        if not all(contant is not None for contant in [uid, data, fmt]):
            raise ServiceError(ResponseStatus.ERROR_REQUEST_NOT_JSON)
        
        __kwargs = {}

        __kwargs['uid'] = uid
        
        if data is not None:
            __kwargs['data'] = RequestParser.convert2image(data)
            
        __kwargs['fmt'] = fmt
                
        if param is not None:
            __kwargs['param'] = param

        return RequestParser(**__kwargs)


@dataclass
class ServerResponse:
    """服务的响应接口"""
    
    code: Optional[int] = 0 
    message: Optional[str] = None 
    uid: Optional[str] = None
    version: Optional[str] = None
    result: Optional[str] = None

