from django.shortcuts import render
from .extract import process_file

def home(request):
    if request.method == 'POST' and request.FILES['file']:
        uploaded_file = request.FILES['file']
        keywords = process_file(uploaded_file)
        return render(request, 'home.html', {'keywords': keywords})
    return render(request, 'home.html')

# Create your views here.
