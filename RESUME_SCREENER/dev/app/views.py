from django.shortcuts import render
from .screener import handle_file_upload, pred

def home(request):
    prediction = None
    if request.method == 'POST' and request.FILES['resume']:
        uploaded_file = request.FILES['resume']
        resume_text = handle_file_upload(uploaded_file)
        prediction = pred(resume_text)
    return render(request, 'home.html', {'prediction': prediction})
