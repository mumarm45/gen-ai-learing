from rag_answer import rag_answer
def chat_answer():     
    while True:
        query = input("Question: ")
    
        if query.lower() in ["quit", "exit", "bye"]:
            print("Answer: Goodbye!")
            break

        result = rag_answer(query)
        print("Answer:", result)
    return result

if __name__ == "__main__":
    chat_answer() 