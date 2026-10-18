import os

from flask import Flask, render_template, request, redirect, url_for, session
import requests
import markdown
from markupsafe import Markup

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")

AGENT_URL = "https://api.perplexity.ai/v1/agent"
AGENT_PRESET = os.environ.get("PERPLEXITY_PRESET", "low")


def ask_agent(question):
    """Ask the Perplexity Agent API; return (answer_markdown, sources)."""
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        raise RuntimeError("PERPLEXITY_API_KEY is not set")

    res = requests.post(
        AGENT_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "preset": AGENT_PRESET,
            "instructions": "Be precise and concise.",
            "input": question,
        },
        timeout=120,
    )
    if res.status_code == 429:
        retry = res.headers.get("Retry-After", "a few")
        raise RuntimeError(f"Rate limited by Perplexity; retry in {retry} seconds")
    if res.status_code == 401:
        raise RuntimeError("Perplexity rejected the API key (401)")
    res.raise_for_status()

    text_parts, sources = [], {}
    for item in res.json().get("output", []):
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") != "output_text":
                continue
            text_parts.append(part.get("text", ""))
            for ann in part.get("annotations") or []:
                if ann.get("type") == "url_citation" and ann.get("url"):
                    sources.setdefault(ann["url"], ann.get("title") or ann["url"])
    return "\n".join(text_parts), sources


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form.get("prompt", "")
        if user_input:
            try:
                markdown_content, sources = ask_agent(user_input)
                if sources:
                    markdown_content += "\n\n**Sources**\n\n" + "\n".join(
                        f"- [{title}]({link})" for link, title in sources.items()
                    )
                html_content = markdown.markdown(
                    markdown_content,
                    extensions=['fenced_code', 'tables', 'nl2br']
                )
                # Store result in session
                session['result'] = str(html_content)
                session['prompt'] = user_input
            except Exception as e:
                session['result'] = f"Error: {e}"
                session['prompt'] = user_input
        
        # Redirect to prevent resubmission
        return redirect(url_for('index'))
    
    # GET request - display result if available
    result = Markup(session.pop('result', ''))  # pop removes it after displaying
    prompt = session.pop('prompt', '')
    
    return render_template("index.html", result=result, prompt=prompt)


if __name__ == "__main__":
    app.run(debug=True)