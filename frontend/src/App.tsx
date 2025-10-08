import { DeepChat } from "deep-chat-react";
import "./style.css";

function App() {
  return (
    <div className="App">
      <h1>Healthcare Support & FAQ Chatbot</h1>
      <DeepChat
        introMessage={{ text: "Hi, I'm a Medibot.  I can answere your FAQ" }}
        style={{
          borderRadius: "10px",
          backgroundImage: `linear-gradient(rgba(255, 255, 255, 0.6), rgba(255, 255, 255, 0.6)), url('/chat-bg.png')`,
          backgroundSize: "cover",
          backgroundRepeat: "no-repeat",
          backgroundPosition: "center",
        }}
        // audio={true}
        microphone={true}
        connect={{
          url: "http://127.0.0.1:8000/chat/",
          method: "POST",
          headers: { "Content-Type": "application/json" },
        }}
      />
    </div>
  );
}

export default App;
