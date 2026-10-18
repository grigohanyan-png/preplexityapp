import os
import time

from flask import Flask, render_template, request, redirect, url_for, session
import requests
import markdown
from markupsafe import Markup, escape

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")

AGENT_URL = "https://api.perplexity.ai/v1/agent"
AGENT_PRESET = os.environ.get("PERPLEXITY_PRESET", "low")


# (connect, read) seconds; keep the total under gunicorn's --timeout in render.yaml
AGENT_TIMEOUT = (10, 90)
AGENT_RETRIES = 2  # extra attempts after the first one
RETRY_STATUSES = {429, 500, 502, 503, 504}


class AgentError(RuntimeError):
    """An API failure with a message that is safe to show to the user."""


def _retry_delay(attempt, res=None):
    """Seconds to wait before the next attempt: Retry-After if short, else backoff."""
    if res is not None:
        try:
            return min(float(res.headers.get("Retry-After", "")), 10)
        except ValueError:
            pass
    return 2 ** attempt


def _post_agent(api_key, payload):
    """POST to the Agent API, retrying connection errors, 429 and 5xx."""
    for attempt in range(AGENT_RETRIES + 1):
        last = attempt == AGENT_RETRIES
        try:
            res = requests.post(
                AGENT_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=AGENT_TIMEOUT,
            )
        except requests.exceptions.ConnectTimeout:
            if last:
                raise AgentError("Could not reach Perplexity (connection timed out). Please try again.")
        except requests.exceptions.ReadTimeout:
            # Not retried: another full wait would outlast the worker timeout.
            raise AgentError("Perplexity took too long to answer. Try a shorter question or try again later.")
        except requests.exceptions.ConnectionError:
            if last:
                raise AgentError("Could not connect to Perplexity. Check your connection and try again.")
        else:
            if res.status_code not in RETRY_STATUSES or last:
                return res
            time.sleep(_retry_delay(attempt, res))
            continue
        time.sleep(_retry_delay(attempt))


def ask_agent(question):
    """Ask the Perplexity Agent API; return (answer_markdown, sources)."""
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        raise AgentError("The server is missing PERPLEXITY_API_KEY. Ask the site owner to set it.")

    res = _post_agent(api_key, {
        "preset": AGENT_PRESET,
        "instructions": "Be precise and concise.",
        "input": question,
    })
    if res.status_code == 429:
        retry = res.headers.get("Retry-After", "a few")
        raise AgentError(f"Too many requests to Perplexity right now. Please retry in {retry} seconds.")
    if res.status_code in (401, 403):
        raise AgentError("Perplexity rejected the API key. Ask the site owner to check it.")
    if res.status_code >= 500:
        raise AgentError(f"Perplexity is having problems ({res.status_code}). Please try again in a minute.")
    if res.status_code >= 400:
        app.logger.warning("Perplexity %s: %s", res.status_code, res.text[:500])
        raise AgentError(f"Perplexity could not process this request ({res.status_code}).")
    try:
        data = res.json()
    except ValueError:
        raise AgentError("Perplexity returned an unreadable response. Please try again.")

    text_parts, sources = [], {}
    for item in data.get("output", []):
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
            except AgentError as e:
                session['result'] = f"Error: {escape(str(e))}"
                session['prompt'] = user_input
            except Exception:
                app.logger.exception("Unexpected error answering a question")
                session['result'] = "Error: something went wrong on our side. Please try again."
                session['prompt'] = user_input
        
        # Redirect to prevent resubmission
        return redirect(url_for('index'))
    
    # GET request - display result if available
    result = Markup(session.pop('result', ''))  # pop removes it after displaying
    prompt = session.pop('prompt', '')
    
    return render_template("index.html", result=result, prompt=prompt)


if __name__ == "__main__":
    app.run(debug=True)