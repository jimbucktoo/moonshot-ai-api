from flask import Flask, request, jsonify
import httpx
import time
import re
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import psutil
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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",
    http_client=httpx.Client(
        timeout=30.0,
        limits=httpx.Limits(
            max_keepalive_connections=5,
            max_connections=10
        )
    )
)

system_message = """
You are an evaluator of startup business ideas from a startup accelerator - MoonshotAI...
(Full system message with evaluation criteria goes here)
"""

RETRY_EXCEPTIONS = (
    httpx.RemoteProtocolError,
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    httpx.WriteTimeout,
    httpx.RequestError
)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    before_sleep=lambda _: print("Connection issue, retrying...")
)
def evaluate_criteria(proposal):
    user_message = f"\nThis is the solution that you are going to evaluate: \n {proposal} \n"
    try:
        time.sleep(max(0, 1.2 - (time.time() % 1)))
        completion = client.chat.completions.create(
            model="deepseek-ai/deepseek-r1",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            stream=True,
            timeout=httpx.Timeout(30.0, read=300.0)
        )

        response_text = []
        last_activity = time.time()
        chunk_count = 0

        for chunk in completion:
            if chunk_count % 5 == 0 and psutil.virtual_memory().percent > 75:
                print("[WARN] Memory usage high, limiting response size")
                response_text = response_text[-1000:]

            if time.time() - last_activity > 10:
                print("Receiving data...")
                last_activity = time.time()

            if chunk.choices[0].delta.content:
                response_text.append(chunk.choices[0].delta.content)
                chunk_count += 1

        return "".join(response_text)

    except Exception as e:
        print(f"Request failed: {type(e).__name__} - {str(e)}")
        raise

def extract_key_elements(text):
    overall_score_match = re.search(r"Overall average score:\s*(\d+(.\d+)?)", text)
    overall_score = overall_score_match.group(1) if overall_score_match else None

    criteria_pattern = re.findall(
        r"The score of criteria(\d+):\s*(\d+)\s*\n"
        r"Summary reasoning criteria\1:\s*(.*?)\n\n"
        r"Improvement suggestion criteria\1:\s*(.*?)\n\n",
        text, re.DOTALL
    )

    extracted_data = {"overall_score": overall_score}
    for num, score, summary, improvement in criteria_pattern:
        num = int(num)
        extracted_data[f"score_criteria{num}"] = int(score)
        extracted_data[f"summary_reasoning_criteria{num}"] = summary.strip()
        extracted_data[f"improvement_suggestion_criteria{num}"] = improvement.strip()

    return extracted_data

@app.route("/evaluate", methods=["POST"])
def evaluate_startup():
    data = request.get_json()
    proposal = data.get("proposal", "")
    if not proposal:
        return jsonify({"error": "Proposal is required"}), 400
    
    response_text = evaluate_criteria(proposal)
    extracted_data = extract_key_elements(response_text)
    return jsonify(extracted_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
