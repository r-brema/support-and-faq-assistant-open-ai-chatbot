import { DeepChat } from 'deep-chat-react';
import './style.css';

function App() {
  return (
    <div className="App">
      <h1>Healthcare Support & FAQ Chatbot</h1>
      <DeepChat
        connect={{
          url: 'http://127.0.0.1:8000/chat/',
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          additionalBodyProps: { question: '{{input_text}}' },
        }}
      />
    </div>
  );
}

export default App;
