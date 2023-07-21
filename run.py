import os
import sys
import dataclasses
import uvicorn
from fastapi import FastAPI
from fastapi import Request
from asgiref.sync import sync_to_async
from werkzeug.exceptions import HTTPException
from loguru import logger
from pathlib import Path

ROOT = Path(__file__).parent
if ROOT not in sys.path:
    sys.path.append(ROOT)
    
from core.utils import * 
from common import * 

logger.remove()
if sys.stdout is not None:
    logger.add(sys.stdout, 
               format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - <level>{message}</level>", 
               backtrace=True, 
               diagnose=True,
               level='DEBUG')
    
logger.add(str(ROOT / 'logs/{time:YYYY-MM-DD-HH-mm}.log'), 
           format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - <level>{message}</level>", 
           rotation='20MB', 
           backtrace=True, 
           diagnose=True,
           level='DEBUG')

app = FastAPI()
ARCH = 'tesorrt'
ARCH = 'onnx'
if ARCH == 'onnx':
    from core.onnx_detector import Detector
    detector = Detector('./weights/fall_detect.onnx')
else:
    import pycuda.autoinit
    from core.tensorrt_detector import TrtModel
    detector = TrtModel('./weights/fall_detect.engine', cuda_ctx=pycuda.autoinit.context)

@app.post('/v1/object-detection')
async def main(request: Request):
    logger.info('starting inference ...')
    try:
        content_type = request.headers.get('Content-Type')
        if content_type is None or content_type.lower() != "application/json":
            raise ServiceError(ResponseStatus.ERROR_REQUEST_NOT_JSON, 'content-type error')

        content = await request.json()

        args = RequestParser.parse_request(content)

        result = await sync_to_async(detector)(args.data)
        print(result)

    except ServiceError as e:
        response_arg = ServerResponse(
            uid = "",
            code=e.code,
            message=e.message,
            result=None)
        return dataclasses.asdict(response_arg)
    except HTTPException as e:
        response_arg = ServerResponse(
            uid = "",
            code=e.code,
            message=f'{e.name}: {e.description}',
            result=None)
        return dataclasses.asdict(response_arg)
    except Exception as e:
        response_arg = ServerResponse(
            uid = "",
            code=ResponseStatus.ERROR_UNKNOWN.code,
            message=ResponseStatus.ERROR_UNKNOWN.message,            
            result=None,
            version='0.0.1')
        return dataclasses.asdict(response_arg)
    
    response_arg = ServerResponse(
        code=ResponseStatus.SUCCESS.code,
        message=ResponseStatus.SUCCESS.message,
        uid = args.uid,
        result=result,
        version='0.0.1')
    response_arg = dataclasses.asdict(response_arg)
    return response_arg


async def run_server():
    config = uvicorn.Config(app=app, host='0.0.0.0', port=10086, reload=False)
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == '__main__':
    # uvicorn run:app --reload
    # app.run(host='0.0.0.0', port=2222, debug=True)

    # uvicorn.run(app='run:app', host='0.0.0.0', port=10086, reload=False)

    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_server())
