import random
def generate_response(prompt_type, user_input):
    if prompt_type == "question":
        return random.choice([
            "Paris is the capital of France.",
            "The Eiffel Tower is a symbol of France, built in 1889.",
            "Paris is known for its culture, food, and landmarks."
        ])
    elif prompt_type == "summary":
        return "This is a brief summary of the given text: " + user_input[:75] + "..."
    elif prompt_type == "creative":
        return random.choice([
            "Once upon a time, a dragon befriended a princess in a forest kingdom...",
            "In the golden hues of autumn, leaves danced like fire in the wind.",
            "A boy discovered a portal to another universe hidden in his attic."
        ])
    return "I'm not sure how to respond."

def feedback_loop(response):
    print(f"\nAI Response: {response}")
    feedback = input("Was this response helpful? (yes/no): ").strip().lower()
    if feedback == "yes":
        print("Thanks for your feedback!")
    else:
        print("Thanks! We'll work to improve it.")

def main():
    while True:
        print("\nWelcome to Your AI Assistant")
        print("1. Answer a factual question")
        print("2. Summarize a text")
        print("3. Generate creative content")
        print("4. Exit")
        choice = input("Select an option (1-4): ").strip()

        if choice == '1':
            question = input("Enter your factual question: ")
            prompt_variants = [
                f"What is the answer to: {question}?",
                f"Can you tell me more about: {question}?",
                f"Give me three facts about: {question}."
            ]
            response = generate_response("question", random.choice(prompt_variants))
            feedback_loop(response)

        elif choice == '2':
            text = input("Paste the text you want summarized: ")
            prompt_variants = [
                f"Summarize the following: {text}",
                f"Briefly explain what this is about: {text}",
                f"What are the main points of this text: {text}"
            ]
            response = generate_response("summary", random.choice(prompt_variants))
            feedback_loop(response)

        elif choice == '3':
            idea = input("What type of creative content do you want (story, poem, idea)? ")
            prompt_variants = [
                f"Write a short {idea} involving a robot and a child.",
                f"Create a {idea} about nature and memory.",
                f"Generate a {idea} set in the future on Mars."
            ]
            response = generate_response("creative", random.choice(prompt_variants))
            feedback_loop(response)

        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid input. Please try again.")

if __name__ == "__main__":
    main()
