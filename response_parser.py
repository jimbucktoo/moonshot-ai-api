import re
import json
import os

def parse_agent_response(response):
    """Structurally parse the API response"""
    result = {
            "final_answer": "",
            "execution_steps": [],
            "search_links": []
            }

    current_step = {}
    link_pattern = re.compile(r'https?://\S+')
    # Add detailed result matching pattern
    detail_pattern = re.compile(r'### 2\. Task outcome \(extremely detailed version\):(.+?)(?=### 3|\Z)', re.DOTALL)

    for msg in response:
        content = str(msg.get('content', ''))
        metadata = msg.get('metadata', {})

        # Parse the final answer
        if "Final answer:" in content:
            result["final_answer"] = _extract_final_answer(content)

        # Parse execution steps (only process assistant messages)
        if msg.get('role') == 'assistant' and content.startswith("**Step"):
            current_step = {
                    "title": content.strip('* '),
                    "details": []
                    }
            result["execution_steps"].append(current_step)
        elif current_step and msg.get('role') == 'assistant':
            # Precisely extract detailed analysis content
            if (detail_match := detail_pattern.search(content)):
                cleaned_content = re.sub(r'\*{2,}|`{3,}', '',
                                         detail_match.group(1)).strip()
                current_step["details"].append(cleaned_content)

        # Extract links from execution logs
        if metadata.get('title') == '📝 Execution Logs':
            result["search_links"].extend(
                    link_pattern.findall(content)
                    )

    return result

def _extract_final_answer(content):
    """Extract and clean the final answer, convert dictionary format to Markdown"""
    # Extract content after "Final answer:"
    answer = content.split("Final answer:")[-1].strip()
    # Remove asterisks and other markers
    answer = re.sub(r'\*{2,}', '', answer).strip()

    # Check if the content is in dictionary format
    dict_match = re.search(r'\{.*\}', answer, re.DOTALL)
    if dict_match:
        try:
            # Attempt to extract novelty_score and report
            score_match = re.search(r"'novelty_score':\s*(\d+)", answer)
            report_match = re.search(r"'report':\s*'(.*?)'(?=\}$|\}\s*$)", answer, re.DOTALL)

            if score_match and report_match:
                # Extract score
                score = f"# Novelty Score: {score_match.group(1)}/100"

                # Extract and clean report content
                report = report_match.group(1)
                # Handle escape characters
                report = report.replace('\\n', '\n').replace("\\'", "'").replace('\\"', '"')

                # Return formatted Markdown
                return f"{score}\n\n{report}"
        except Exception:
            pass

    # Return the original content if not in dictionary format or parsing fails
    return answer

def save_as_markdown(result, filename):
    """Generate an optimized Markdown report"""
    with open(filename, 'w', encoding='utf-8') as f:
        # Final answer section
        f.write("## MoonshotAI: Your Startup Novelty Deep Research Report\n")
        f.write(result["final_answer"] + "\n")

        # Research steps section - Add step deduplication
        if result["execution_steps"]:
            f.write("\n### Execution Steps\n")
            seen_steps = set()
            unique_steps = []

            for step in result["execution_steps"]:
                # Use title and details as uniqueness criteria
                step_content = (step['title'], tuple(step['details']))
                if step_content not in seen_steps:
                    seen_steps.add(step_content)
                    unique_steps.append(step)

            # Output deduplicated steps
            for step in unique_steps:
                f.write(f"\n#### {step['title']}\n")
                f.write('\n'.join(step["details"]) + "\n")

        # Search results section
        if result["search_links"]:
            f.write("\n### Relevant References\n")
            seen = set()
            for raw_link in result["search_links"]:
                # More robust link cleaning logic
                # 1. Remove URL parameters
                link = raw_link.split('?')[0]
                # 2. Handle extra content after links (including parentheses and newlines)
                link = re.sub(r'\).*$', '', link)
                # 3. Clean various special characters
                link = re.sub(r'[\\n\s.]+$', '', link)
                # 4. Handle Markdown format in links
                link = re.sub(r'\]\(.*$', '', link)

                if link and link.startswith('http') and link not in seen:
                    seen.add(link)
                    f.write(f"- {link}\n")

if __name__ == "__main__":
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct file paths using relative paths
    json_path = os.path.join(current_dir, "last_result.json")
    report_path = os.path.join(current_dir, "analysis_report.md")

    # Load data from the local JSON file
    with open(json_path, encoding="utf-8") as f:
        test_data = json.load(f)

    parsed_result = parse_agent_response(test_data)
    save_as_markdown(
            parsed_result,
            report_path
            )
    print(f"Complete analysis report saved to: {report_path}")
