import json
import os
import tempfile
import uuid

from flask import Flask, render_template, request, redirect, url_for, session
import requests
import markdown
import nh3
from markupsafe import Markup, escape

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")

AGENT_URL = "https://api.perplexity.ai/v1/agent"
AGENT_PRESET = os.environ.get("PERPLEXITY_PRESET", "low")

# Results live on disk, not in the cookie session (cookies cap out near 4 KB).
# The temp dir is shared by all gunicorn workers on the instance.
RESULTS_DIR = os.path.join(tempfile.gettempdir(), "preplexity-results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_result(prompt, html):
    result_id = uuid.uuid4().hex
    with open(os.path.join(RESULTS_DIR, result_id + ".json"), "w") as f:
        json.dump({"prompt": prompt, "result": html}, f)
    return result_id


def pop_result(result_id):
    """Load a stored result once and delete it; return {} if missing."""
    if not result_id or not result_id.isalnum():
        return {}
    path = os.path.join(RESULTS_DIR, result_id + ".json")
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    try:
        os.remove(path)
    except OSError:
        pass
    return data


# Model output is untrusted: keep only the tags Markdown produces, and force
# links to open safely in a new tab.
ALLOWED_TAGS = {
    "p", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6",
    "strong", "em", "b", "i", "del", "code", "pre", "blockquote",
    "ul", "ol", "li", "a",
    "table", "thead", "tbody", "tr", "th", "td",
}
ALLOWED_ATTRIBUTES = {"a": {"href", "title"}, "th": {"align"}, "td": {"align"}}


def render_markdown(text):
    html = markdown.markdown(text, extensions=['fenced_code', 'tables', 'nl2br'])
    html = nh3.clean(
        html,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        url_schemes={"http", "https", "mailto"},
        link_rel="noopener noreferrer nofollow",
        set_tag_attribute_values={"a": {"target": "_blank"}},
    )
    return html


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
                html_content = render_markdown(markdown_content)
            except Exception as e:
                html_content = str(escape(f"Error: {e}"))
            session['result_id'] = save_result(user_input, str(html_content))
        
        # Redirect to prevent resubmission
        return redirect(url_for('index'))
    
    # GET request - display result if available
    data = pop_result(session.pop('result_id', None))
    result = Markup(data.get('result', ''))
    prompt = data.get('prompt', '')
    
    return render_template("index.html", result=result, prompt=prompt)


if __name__ == "__main__":
    app.run(debug=True)