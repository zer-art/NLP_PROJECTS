from django.shortcuts import render
from . import suggest

def home(request):
    suggestions = None
    error = None
    if request.method == "POST":
        word = request.POST.get("word")
        if word:
            try:
                suggestions_df = suggest.autocorrect(word)  # Get suggestions as a DataFrame
                suggestions = suggestions_df["word"].tolist()  # Convert to a list of words
            except Exception as e:
                error = f"Error generating suggestions: {e}"
        else:
            error = "Please enter a valid word."
    return render(request, "home.html", {"suggestions": suggestions, "error": error})
