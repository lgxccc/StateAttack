import httpx
import json
import time
from openai import OpenAI

def create_conversation_prompt(first_round):

    prompt = [
        {
            "role": "system",
            "content": """You are an expert at creating natural follow-up conversations. Your task is to generate exactly 2 new conversation rounds that naturally follow from a given first exchange.

Task: Generate 2 new conversation rounds (4 messages total) that naturally follow the provided first exchange.

Requirements:
1. Generate exactly 4 messages: user1, assistant1, user2, assistant2
2. Both user messages MUST start with "why"
3. The questions should seek clarification, reasoning, or deeper understanding of topics from the first exchange
4. Assistant responses should be comprehensive, natural, and directly answer the "why" questions
5. All conversations must be in English
6. The conversations should feel like natural follow-ups that someone might ask after hearing the first assistant response

Output format: Return a JSON object with "conversations" array containing exactly 4 objects with "role" and "content" fields.

Example output structure:
{
    "conversations": [
        {"role": "user", "content": "Why did you choose..."},
        {"role": "assistant", "content": "I chose that approach because..."},
        {"role": "user", "content": "Why is this method..."},
        {"role": "assistant", "content": "This method is effective because..."}
    ]
}"""
        },
        {
            "role": "user",
            "content": f"""Based on the following first exchange, generate 2 new conversation rounds where the user asks "why" questions related to the assistant's response:

First exchange:
User: {first_round[0]['content']}
Assistant: {first_round[1]['content']}

Please generate 2 natural follow-up rounds where the user asks insightful "why" questions about the assistant's response, and the assistant provides detailed explanations."""
        }
    ]
    return prompt


def extract_first_round(conversation):
    convs = conversation["conversations"]
    if len(convs) >= 2 and convs[0]["role"] == "user" and convs[1]["role"] == "assistant":
        return [convs[0], convs[1]]
    else:
        raise ValueError("对话格式不正确：无法找到完整的第一轮对话")


def extract_remaining_rounds(conversation):
    convs = conversation["conversations"]
    return convs[2:] if len(convs) > 2 else []


def combine_conversations(first_round, generated_rounds, remaining_rounds):
    all_conversations = first_round + generated_rounds + remaining_rounds
    return {"conversations": all_conversations}


def validate_generated_rounds(generated_data):
    try:
        convs = generated_data["conversations"]

        if len(convs) != 4:
            return False, f"应该生成4条消息，实际生成了{len(convs)}条"

        expected_roles = ["user", "assistant", "user", "assistant"]
        for i, expected_role in enumerate(expected_roles):
            if convs[i]["role"] != expected_role:
                return False, f"第{i + 1}条消息角色错误，期望{expected_role}，实际{convs[i]['role']}"

        # 检查user消息是否以why开头
        if not convs[0]["content"].strip().lower().startswith("why"):
            return False, "第1条user消息未以why开头"
        if not convs[2]["content"].strip().lower().startswith("why"):
            return False, "第3条user消息未以why开头"

        return True, "验证通过"
    except Exception as e:
        return False, f"验证时出错: {e}"


def extract_json_from_response(content):
    """从API响应中提取JSON内容"""
    # 处理可能的markdown代码块
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    return content


def process_single_conversation(client, conversation, index):
    """处理单个对话"""
    try:
        print(f"Processing conversation {index + 1}...")

        # 提取第一轮对话
        first_round = extract_first_round(conversation)
        remaining_rounds = extract_remaining_rounds(conversation)

        # 构造prompt
        prompt = create_conversation_prompt(first_round)

        # 调用API
        response = client.chat.completions.create(
            model="gpt-4o-mini",  #gpt-4o-mini    o4-mini
            messages=prompt,
            temperature=0.7
        )
        usage = response.usage
        total_tokens = usage.total_tokens

        content = response.choices[0].message.content.strip()

        # 提取JSON
        json_content = extract_json_from_response(content)

        # 解析生成的对话
        generated_data = json.loads(json_content)

        # 验证生成结果
        is_valid, message = validate_generated_rounds(generated_data)

        if is_valid:
            # 组合所有对话
            generated_rounds = generated_data["conversations"]
            final_conversation = combine_conversations(first_round, generated_rounds, remaining_rounds)

            print(f"✓ Successfully processed conversation {index + 1}")
            print(f"  Generated Q1: {generated_rounds[0]['content'][:50]}...")
            print(f"  Generated Q2: {generated_rounds[2]['content'][:50]}...")

            return final_conversation, True, None
        else:
            print(f"✗ Validation failed for conversation {index + 1}: {message}")
            return conversation, False, message

    except Exception as e:
        print(f"✗ Error processing conversation {index + 1}: {e}")
        return conversation, False, str(e)


def process_all_conversations(client, data, delay=0.5):

    successful_count = 0
    failed_conversations = []
    processed_data = data.copy()  # 创建副本避免修改原数据

    total_conversations = len(data)

    for i, conversation in enumerate(data):
        # 处理单个对话
        result_conversation, success, error_msg = process_single_conversation(client, conversation, i)

        if success:
            processed_data[i] = result_conversation
            successful_count += 1
        else:
            failed_conversations.append({
                'index': i + 1,
                'error': error_msg
            })

        if i < total_conversations - 1:  # 最后一个不需要延迟
            time.sleep(delay)

    stats = {
        'total': total_conversations,
        'successful': successful_count,
        'failed': len(failed_conversations),
        'success_rate': (successful_count / total_conversations) * 100,
        'failed_details': failed_conversations
    }

    # 打印结果
    print(f"\n{'=' * 50}")
    print(f"处理完成！")
    print(f"总对话数: {stats['total']}")
    print(f"成功处理: {stats['successful']}")
    print(f"失败数量: {stats['failed']}")
    print(f"成功率: {stats['success_rate']:.1f}%")
    print(f"预计token节省: 60-80%")

    if failed_conversations:
        print(f"\n失败的对话:")
        for fail in failed_conversations:
            print(f"  对话 {fail['index']}: {fail['error']}")

    return processed_data, stats

def load_data(filename):
    """从文件加载数据"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ 成功加载数据: {filename}")
        return data
    except Exception as e:
        print(f"✗ 加载数据失败: {e}")
        return None

client = OpenAI(
        base_url="https://svip.xty.app/v1",
        api_key="sk-yqA5ril4wMxW5wFMAd31C37bF2C544098eAc0fD434Db98Bf",
        http_client=httpx.Client(
            base_url="https://svip.xty.app/v1",
            follow_redirects=True,
        ),
    )

with open("../modify_dataset/ultrachat_2000.json", "r", encoding="utf-8") as f1:
    data_instruct = json.load(f1)

data = data_instruct[:200]  # 取出前200个
del data_instruct[:200]
processed_data, stats = process_all_conversations(client, data)

with open("ultrachat_200_2why.json", "w", encoding="utf-8") as f2:
    json.dump(processed_data, f2, indent=2, ensure_ascii=False)

with open("ultrachat_1800.json", "w", encoding="utf-8") as f3:
    json.dump(data_instruct, f3, indent=2, ensure_ascii=False)