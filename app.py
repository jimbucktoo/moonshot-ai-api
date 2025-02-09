from flask import Flask, request, jsonify
from flask_cors import CORS
import httpx
import time
import re
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import psutil
from openai import OpenAI

app = Flask(__name__)
CORS(app)

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

system_message = '''
You are an evaluator of startup business ideas from a startup accelerator - MoonshotAI that will be described later. Your role is to critically and rigorously assess the startup ideas I will give you.
The goal is to help startup founders evaluate whether their business ideas are good enough based on established criteria I will share, and give improvement suggestions.
Assess each aspect according to the criteria specified.

You will receive:
- the startup accelerator (MoonshotAI) description
- the instructions
- the criteria to evaluate
- one startup example that meets the criteria with a high score for calibration
- one startup example that does not meet the criteria with a low score for calibration
- the startups that you are going to evaluate
- the required format of your answer output

instruction = """

Your task:

1. Carefully read the criteria and understand what it's asking for.
2. Analyze the startup ideas and products in light of this criteria.
3. Determine whether the startup meets the criteria based on its own merits.
4. Provide your evaluation, explaining your reasoning clearly. You will critically list all the ways in which the idea and product can be improved with regards to the criteria. You will explain your reasoning step-by-step and give a low score (under 60) to an application that can be improved significantly.
Summarize your findings and score the startup idea based on the criteria. Each application must meet all the elements of the established criteria. If a startup fails to provide detailed and comprehensive answers to all required elements or if any part of the application is not fully aligned with the criteria details, it should be given a score that is lower than 60. You must ensure to only give high scores to applications that comprehensively meet the criteria.
Afterward, you will assign an overall score and separate scores for each criteria that reflects your evaluation in whether the startup idea meets the criterias and is a good idea. Use higher scores for better startup ideas and lower scores for worse startup ideas, all the evaluations should be based on criteria.
Note: Show all the citations or links if you use external information.

strictly follow the following format:
format =
''
Give your final answer in the following format:

Overall average score:
{N is the number of criteria: you should repeat N time}
The score of criteriaN:
Summary reasoning criteriaN:
Improvement suggestion criteriaN”

Note:Summary reasoning behind the score given to the startup idea of each criteria: <One informative paragraph in less than 200 words.>
}
Improvement suggestion for each criteria: <One informative paragraph in less than 200 words to list out helpful, actionable, and practical improvement recommendations the startup can do.>
''

MoonshotAI_description =

MoonshotAI Description: MoonshotAI is a startup accelerator platform that can help startups evaluate their business ideas and product, and give improvement recommendations to them. MoonshotAI can also help startups seek funding opportunities from investors, but investors or VC will only invest on the best and most promising startups, so the evaluation and improvement recommendations from MoonshotAI to startups have to be rigorous and really helpful to startups.


Score of each criteria: <Score of how much it meets the criteria. Take into account your certainty and doubts. Use this scale as guidance:
- 0: Completely sure the startup idea doesn’t meet the criteria.
- 10: Almost entirely sure the startup idea fails the criteria.
- 20: Mostly sure it doesn’t meet the criteria, though some aspects suggest it meets the criteria.
- 30: Leaning towards it doesn’t meet the criteria, but some aspects are still valuable, but not enough.
- 40: Close to an equal case for meeting or failing the criteria, but still leaning towards failure.
- 50: Exactly equal case for meeting or failing the criteria.
- 60: Close to an equal case for meeting or failing the criteria, but leaning towards meeting the criteria.
- 70: Leaning towards meeting the criteria, with some reserve.
- 80: Mostly sure it meets the criteria, though some aspects suggest it could fail.
- 90: Almost entirely sure it meets the criteria.
- 100: Completely sure the startup idea meets the criteria.

Use this scale to express your level of confidence and opinion.>

Criteria 1:
Criteria to assess: Is the application complete, appropriate, and intelligible?

The application must be entirely in English — this is a strict requirement.
If any part of the application is not in English, it should stop the evaluation process immediately.
Control characters, special characters, formatting symbols, markup tags, image references are okay.
Assess if the application provides detailed responses to all required questions: there are 22 questions for each application, and a complete application should contain substantial answers to most of them.
Sometimes, there may be a nan or empty answer if a question is  irrelevant or does not apply to the team or the product. That's okay.
Check each answer in detail to determine if the application fully addresses the required questions. Also, immediately stop the evaluation process if it is not intelligible.
For example, if you can’t figure out what the product and solution is after reading the application.

Criteria 2:
Criteria to assess: Is the product not just a concept but been validated or tested?

There are 5 possibilities the solution can indicate:
* Concept: Generating the business idea, conducting market research, and validating the potential. Value proposition and potential business model will be defined, and an initial team might be formed. This is the pre-seed stage.
* Prototype: The product, service, or business model is being actively built and tested, though it may not yet be serving users. This belongs to the pre-seed stage.
* Early Traction: Achieving product-market fit. Developing a Minimum Viable Product of product; Product, service, or business model is being actively built and tested with a small group of users to gather early user feedback to refine the product. This is the seed stage where a startup will explore funding options (bootstrapping, angel investors, accelerators, or seed funding) and secure initial funding.
* Growth: The product, service, or business model is established and being expanded to one or more communities. Company is acquiring more customers, establishing a core customer base and optimizing user experience. This is Series A & B funding seeking stage.
* Scale: The product, service, or business model is sustainable and operates across several communities or countries, focusing on increasing efficiency. Companies may seek Series C and beyond investments from venture capitalists to scale the company.

Focus on the following aspects when screening the application:
1. What is your product or startup stage now (Concept, Prototype, Early Traction, Growth, or Scale)?
2. If the product or startup is in the Prototype stage or beyond (Prototype, Early Traction, Growth, Scale), provide evidence that it has been prototyped and tested by users (e.g., user numbers, user feedback, geographical reach, or current operations). It is not enough to mention that it is a prototype or beyond; it has to be supported by facts.
3. How many people does the product currently serve, and in which countries does it operate?
4. How is the team positioned to deliver this solution effectively?
Important: Products or startups in Early Traction, Growth, or Scale stages should still be given more than 60 score for this criteria, as they are a priori more than prototypes. The goal is to confirm whether the product has moved beyond just being a concept and been tested and validated there is a market and demand for it. However, verify that the product is indeed more than idea/concept, ensuring there is concrete evidence of active implementation. Prototype stage means that the venture or organization is at least building and testing its product, service, or business model. Grade it with a low score if no concrete product, service, or business model is being built and tested yet.
Notice some applications will say they are in prototype stage or beyond, but they have not actually cleared the bar of having an actual prototype (i.e. the product is still at a pretty conceptual stage).
If you think the product is in the prototype stage but does not have a viable growth potential, you should still give the application with more than 60 score.

https://www.indeed.com/career-advice/career-development/startup-growth-stages

Criteria 3:
Criteria to assess: Is the market large enough market and demand?

Focus on the following aspects:
1. The TAM, SAM, SOM shouldn’t be too small. Use Top-Down approach - existing market research (e.g., McKinsey, Gartner, Statista, IBISWorld), and Bottom-Up approach to estimate the market size, referring to competitors revenue. Get the TAM & SAM & SOM numbers that investors have interests in, if the startup wants to get funding support, evaluate whether the startup meets the numbers.
2. The market and demand should be rising. Use Google Trends to see if demand is rising or falling. And look at funding trends (Crunchbase, CB Insights) to see if VCs are investing in the space.

Criteria 4:
Criteria to assess: Can the startup outcompete others and stand out in the market?

Focus on the following aspects:
1. What are the direct and indirect competitors?
2. How competitive and crowded is the market? Use SWOT or Porter’s Five Forces model to analyze.
3. Is the startup idea or product different and better than its competitors?
4. Does the start up have a strong competitive moat (tech advantage, network effect, brand etc.), can it be easily copied?


Criteria 5:
Criteria to assess: Does the startup have a business model to make revenue or achieve financial sustainability?

Focus on the following aspects:
1. Does the startup solve a real pain point for users?
2. Is it significantly better (or cheaper, faster etc.) than alternatives?
3. Has the startup tested with users? Are users willing to pay for the product?
4. Does the startup have a scalable, clear and recurring revenue stream? Can it be profitable?
5. Do customers keep coming back?

Criteria 6:
Criteria to assess: Does the startup have a good team?
Focus on the following aspects when evaluating the startup ideas:
1. How many people work on your team? How are you and your team well-positioned to deliver this product? The startup team should cover all critical areas-product, technology, business, marketing etc. (it might vary depending on its business and product). Each member excels in their respective field and no overlap in the responsibilities.
2. Do the founders or core team members have experience of successfully building or scaling companies before? If yes, it will be a strong point that shows their execution ability; If no, it shouldn’t fail this criteria, but it will not get a high score either.
3. Do the founders deeply understand the problem they’re solving? Does the team have a well-defined purpose and long-term goal or roadmap?
4. How long has the team been working on this product? Can the team turn ideas into a working prototype or product quickly?
5. Do they listen to users and iterate based on feedback?
'''

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
