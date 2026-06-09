from fastapi import APIRouter, UploadFile

api_router = APIRouter()

@api_router.post('/post_images', tags='post_images')
def upload_image():
    pass