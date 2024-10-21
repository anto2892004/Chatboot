import streamlit as st
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

def main():
    groq_api_key = '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'
    system_prompt = "You are a helpful assistant."
    
    spacer, col = st.columns([5, 1])  
    with col:  
        st.image('icon.png')

    st.title("Chat With EcoSmart!")

    model = st.sidebar.selectbox('Choose a model', ['llama3-8b-8192', 'mixtral-8x7b-32768'])
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    user_question = st.text_input("Ask a question:")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    else:
        for message in st.session_state.chat_history:
            memory.save_context({'input': message['human']}, {'output': message['AI']})

    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

    if user_question:
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )

        conversation = LLMChain(
            llm=groq_chat,
            prompt=prompt,
            verbose=True,
            memory=memory,
        )
        
        response = conversation.predict(human_input=user_question)
        message = {'human': user_question, 'AI': response}
        st.session_state.chat_history.append(message)
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
