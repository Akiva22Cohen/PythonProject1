import os
import openai
import json

from AdvancedChatMemory import AdvancedChatMemory
from LifeCoachingBookRetriever import LifeCoachingBookRetriever

from context_intent_analysis import analyze_context_and_intent

from suggest_reply_parts import suggest_reply_part1
from suggest_reply_parts import suggest_reply_part2
from suggest_reply_parts import suggest_reply_part4


# Check if the API key was successfully retrieved
def CheckAPI(api_key):
    if api_key:
        return os.environ['OPENAI_API_KEY'] == api_key


def load_json_from_file(file_path):
    """
    פונקציה שקוראת קובץ JSON ומחזירה את התוכן שלו כאובייקט פייתון.

    :param file_path: נתיב לקובץ JSON
    :return: תוכן הקובץ כאובייקט JSON
    :raises: FileNotFoundError, json.JSONDecodeError
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON in file - {file_path}")
        return None


def main():
    openai.api_key = os.getenv("OPENAI_API_KEY")

    if CheckAPI(openai.api_key):
        print("API key loaded successfully.")
    else:
        print("API key not found in Colab secrets.")

    file_path = './chat_history.json'
    chat_history = load_json_from_file(file_path)

    if chat_history is not None:
        print("JSON loaded successfully!")
    else:
        print("Failed to load JSON.")

    # response = suggest_reply_part1(chat_history)
    # # קבלת התשובה
    # print(response.choices[0].message.content)

    ### Quick Test (Part 2)
    # memory = AdvancedChatMemory(chat_history)

    # reply1 = suggest_reply_part2(memory)
    # print("Reply:", reply1)

    # memory.add_message("user", "Cool, what are you up to later?")
    # reply2 = suggest_reply_part2(memory)
    # print("Reply 2:", reply2)

    # Analyze context and intent
    # analysis = analyze_context_and_intent(chat_history[-5:])
    # print("Analysis:", analysis)

    ###########################################################
    # Quick Test (Part 4)
    ###########################################################
    memory = AdvancedChatMemory()
    book_retriever = LifeCoachingBookRetriever("19\ Emotion\ Coaching\ author\ Media\ \&\ File\ Management.pdf")

    memory.add_message("user", "I'm feeling stuck in my career. Any advice?")
    suggested_reply = suggest_reply_part4(memory, book_retriever)
    print("Model Suggestion:", suggested_reply)


if __name__ == "__main__":
    main()
