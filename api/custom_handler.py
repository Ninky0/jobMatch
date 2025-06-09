from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    custom_errors = []

    for error in errors:
        if "ctx" in error and "pattern" in error["ctx"]:
            custom_errors.append({
                "loc": error["loc"],
                "msg": "사업자등록번호는 ***-**-***** 형식이어야 합니다.",
                "type": error["type"],
                "input": error.get("input", None)
            })
        elif error.get("msg") == "직무 내용은 필수 입력값이며, 10자 이상이어야 합니다.":
            custom_errors.append({
                "loc": error["loc"],
                "msg": "직무 내용은 필수 입력값이며, 10자 이상이어야 합니다.",
                "type": "value_error.min_length",
                "input": error.get("input", None)
            })
        else:
            custom_errors.append(error)

    # JSON 직렬화 문제 해결
    for custom_error in custom_errors:
        if "ctx" in custom_error and "error" in custom_error["ctx"]:
            custom_error["ctx"]["error"] = str(custom_error["ctx"]["error"])

    return JSONResponse(
        status_code=400,
        content={"detail": custom_errors},
    )
