import gradio as gr
import logging
from typing import List, Optional, Tuple

# Import your main system (assuming it's in the same directory)
from latest import PersonalitySystem
from voice_cloning import XTTSVoiceCloner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradioPersonalityInterface:
    def __init__(self):
        """Initialize the Gradio interface with the personality system"""
        try:
            # Initialize voice cloning module
            self.voice_cloner = XTTSVoiceCloner()
            # Initialize personality system with voice cloning
            self.personality_system = PersonalitySystem(voice_cloning_module=self.voice_cloner)

            # Track active conversations for the interface
            self.current_conversation_id = None
            self.current_user_id = None

            logger.info("Gradio interface initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            # Initialize without voice cloning if it fails
            self.voice_cloner = None
            self.personality_system = PersonalitySystem()

    def create_user_profile(self, user_id: str, training_texts: str, reference_audio) -> Tuple[str, str]:
        """Create a new user profile with personality analysis and voice cloning setup"""
        try:
            if not user_id.strip():
                return "❌ Please provide a valid User ID", ""

            if not training_texts.strip():
                return "❌ Please provide training texts for personality analysis", ""

            # Process training texts
            text_list = [text.strip() for text in training_texts.split('\n') if text.strip()]
            if len(text_list) < 5:
                return "❌ Please provide at least 5 lines of training text", ""

            # Create personality profile
            profile = self.personality_system.create_personality_profile(user_id, text_list)

            # Save reference audio if provided
            audio_status = ""
            if reference_audio and self.voice_cloner:
                try:
                    self.personality_system.save_user_reference_audio(user_id, reference_audio)
                    audio_status = "\n🎤 Reference audio saved successfully!"
                except Exception as e:
                    audio_status = f"\n⚠️ Audio processing failed: {str(e)}"
            elif not reference_audio:
                audio_status = "\n⚠️ No reference audio provided - voice cloning will be disabled"

            # Create profile summary
            profile_summary = f"""
✅ **Profile Created Successfully for {user_id}**

**Personality Traits:**
• Openness: {profile.openness:.2f}
• Conscientiousness: {profile.conscientiousness:.2f}
• Extraversion: {profile.extraversion:.2f}
• Agreeableness: {profile.agreeableness:.2f}
• Neuroticism: {profile.neuroticism:.2f}

**Communication Style:**
• Formality Level: {profile.formality_level:.2f}
• Emotional Expressiveness: {profile.emotional_expressiveness:.2f}
• Response Length: {profile.response_length_preference}

**Training Data:** {len(text_list)} text samples processed
{audio_status}
            """

            return f"✅ User profile '{user_id}' created successfully!", profile_summary.strip()

        except Exception as e:
            logger.error(f"Profile creation failed: {e}")
            return f"❌ Profile creation failed: {str(e)}", ""

    def start_conversation(self, user_id: str) -> Tuple[str, str, List]:
        """Start a new conversation with a user"""
        try:
            if not user_id.strip():
                return "❌ Please enter a User ID", "", []

            # Load user profile if not already loaded
            if user_id not in self.personality_system.user_profiles:
                profile = self.personality_system.load_user_profile(user_id)
                if not profile:
                    return f"❌ No profile found for user '{user_id}'. Please create a profile first.", "", []

            # Start new conversation
            conversation_id = self.personality_system.start_conversation(user_id)
            self.current_conversation_id = conversation_id
            self.current_user_id = user_id

            return f"✅ Conversation started with {user_id}", f"Conversation ID: {conversation_id}", []

        except Exception as e:
            logger.error(f"Failed to start conversation: {e}")
            return f"❌ Failed to start conversation: {str(e)}", "", []

    def send_message(self, message: str, include_voice: bool) -> Tuple[List, Optional[str]]:
        """Send a message and get response"""
        try:
            if not self.current_conversation_id:
                return [["System", "❌ Please start a conversation first"]], None

            if not message.strip():
                return [["System", "❌ Please enter a message"]], None

            # Generate response
            response_data = self.personality_system.generate_response(
                self.current_conversation_id,
                message.strip(),
                include_voice=include_voice
            )

            # Get conversation history for display
            history = self.personality_system.get_conversation_history(self.current_conversation_id)

            # Format history for gradio chatbot
            chat_history = []
            for msg in history[-10:]:  # Show last 10 messages
                role = "You" if msg["role"] == "user" else "AI"
                chat_history.append([role, msg["content"]])

            # Return audio path if available
            audio_path = response_data.get("audio_path")

            return chat_history, audio_path

        except Exception as e:
            logger.error(f"Message processing failed: {e}")
            return [["System", f"❌ Error: {str(e)}"]], None

    def provide_feedback(self, rating: float, feedback_type: str) -> str:
        """Provide feedback for the last AI response"""
        try:
            if not self.current_conversation_id:
                return "❌ No active conversation to provide feedback for"

            # Get conversation history
            history = self.personality_system.get_conversation_history(self.current_conversation_id)

            if not history:
                return "❌ No messages to provide feedback for"

            # Find the last AI message
            last_ai_message_index = None
            for i in range(len(history) - 1, -1, -1):
                if history[i]["role"] == "assistant":
                    last_ai_message_index = i
                    break

            if last_ai_message_index is None:
                return "❌ No AI messages found to provide feedback for"

            # Provide feedback
            self.personality_system.provide_feedback(
                self.current_conversation_id,
                last_ai_message_index,
                rating,
                feedback_type
            )

            return f"✅ Feedback submitted: {rating}/5 ({feedback_type})"

        except Exception as e:
            logger.error(f"Feedback submission failed: {e}")
            return f"❌ Feedback submission failed: {str(e)}"

    def get_profile_info(self, user_id: str) -> str:
        """Get information about a user's profile"""
        try:
            if not user_id.strip():
                return "❌ Please enter a User ID"

            # Try to load profile
            if user_id not in self.personality_system.user_profiles:
                profile = self.personality_system.load_user_profile(user_id)
                if not profile:
                    return f"❌ No profile found for user '{user_id}'"
            else:
                profile = self.personality_system.user_profiles[user_id]

            # Format profile information
            info = f"""
**Personality Profile for {user_id}**

**Big Five Traits:**
• Openness: {profile.openness:.2f} (Creativity, curiosity)
• Conscientiousness: {profile.conscientiousness:.2f} (Organization, discipline)
• Extraversion: {profile.extraversion:.2f} (Sociability, assertiveness)
• Agreeableness: {profile.agreeableness:.2f} (Cooperation, trust)
• Neuroticism: {profile.neuroticism:.2f} (Emotional instability)

**Communication Style:**
• Formality Level: {profile.formality_level:.2f}
• Emotional Expressiveness: {profile.emotional_expressiveness:.2f}
• Vocabulary Complexity: {profile.vocabulary_complexity:.2f}
• Preferred Response Length: {profile.response_length_preference}

**Interests:** {', '.join(profile.topics_of_interest[:5]) if profile.topics_of_interest else 'None identified'}
            """

            return info.strip()

        except Exception as e:
            logger.error(f"Profile info retrieval failed: {e}")
            return f"❌ Failed to retrieve profile info: {str(e)}"


def create_interface():
    """Create and configure the Gradio interface"""

    # Initialize the interface
    interface = GradioPersonalityInterface()

    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .tab-nav button {
        font-size: 16px !important;
        padding: 10px 20px !important;
    }
    .feedback-section {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
    }
    """

    with gr.Blocks(css=css, title="AI Personality & Voice Cloning System") as app:
        gr.Markdown("""
        # 🤖 AI Personality & Voice Cloning System

        Create AI personalities that match real people's communication style and voice!
        """)

        with gr.Tabs():
            # Tab 1: Profile Creation
            with gr.TabItem("👤 Create Profile", elem_id="profile-tab"):
                gr.Markdown("### Create a new personality profile")

                with gr.Row():
                    with gr.Column(scale=1):
                        profile_user_id = gr.Textbox(
                            label="User ID",
                            placeholder="Enter unique user identifier (e.g., john_doe)",
                            info="This will be used to identify the personality profile"
                        )

                        training_texts = gr.Textbox(
                            label="Training Texts",
                            placeholder="Paste text samples that represent the person's writing style...\nEach line should be a separate example.\nMinimum 5 examples required.",
                            lines=10,
                            info="Enter text samples (conversations, emails, messages) - one per line"
                        )

                        reference_audio = gr.Audio(
                            label="Reference Audio (Optional)",
                            type="filepath",

                        )

                        create_btn = gr.Button("🚀 Create Profile", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        profile_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            lines=2
                        )

                        profile_details = gr.Markdown(
                            label="Profile Details",
                            value="Profile details will appear here after creation..."
                        )

                create_btn.click(
                    interface.create_user_profile,
                    inputs=[profile_user_id, training_texts, reference_audio],
                    outputs=[profile_status, profile_details]
                )

            # Tab 2: Chat Interface
            with gr.TabItem("💬 Chat", elem_id="chat-tab"):
                gr.Markdown("### Chat with AI personality")

                with gr.Row():
                    with gr.Column(scale=2):
                        chat_user_id = gr.Textbox(
                            label="User ID",
                            placeholder="Enter the user ID to chat with",
                            info="Must match a created profile"
                        )
                        start_chat_btn = gr.Button("🎯 Start Conversation", variant="secondary")
                        chat_status = gr.Textbox(label="Status", interactive=False, lines=1)

                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=400,
                            bubble_full_width=False
                        )

                        with gr.Row():
                            message_input = gr.Textbox(
                                label="Your Message",
                                placeholder="Type your message here...",
                                lines=2,
                                scale=4
                            )
                            include_voice = gr.Checkbox(
                                label="🎤 Include Voice",
                                value=True,
                                info="Generate voice response"
                            )

                        send_btn = gr.Button("📤 Send Message", variant="primary")

                        # Audio output
                        audio_output = gr.Audio(
                            label="AI Voice Response",
                            autoplay=False,
                            visible=True
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Feedback")
                        gr.Markdown("Rate the AI's last response to help improve the model")

                        feedback_rating = gr.Slider(
                            minimum=1,
                            maximum=5,
                            step=1,
                            value=3,
                            label="Rating (1-5)",
                            info="1 = Poor, 5 = Excellent"
                        )

                        feedback_type = gr.Dropdown(
                            choices=["satisfaction", "personality_match", "voice_quality", "helpfulness"],
                            value="satisfaction",
                            label="Feedback Type"
                        )

                        feedback_btn = gr.Button("📝 Submit Feedback", variant="secondary")
                        feedback_status = gr.Textbox(label="Feedback Status", interactive=False, lines=1)

                # Event handlers
                start_chat_btn.click(
                    interface.start_conversation,
                    inputs=[chat_user_id],
                    outputs=[chat_status, gr.Textbox(visible=False), chatbot]
                )

                send_btn.click(
                    interface.send_message,
                    inputs=[message_input, include_voice],
                    outputs=[chatbot, audio_output]
                ).then(
                    lambda: "",  # Clear message input
                    outputs=[message_input]
                )

                # Allow Enter key to send message
                message_input.submit(
                    interface.send_message,
                    inputs=[message_input, include_voice],
                    outputs=[chatbot, audio_output]
                ).then(
                    lambda: "",
                    outputs=[message_input]
                )

                feedback_btn.click(
                    interface.provide_feedback,
                    inputs=[feedback_rating, feedback_type],
                    outputs=[feedback_status]
                )

            # Tab 3: Profile Management
            with gr.TabItem("📋 Manage Profiles", elem_id="manage-tab"):
                gr.Markdown("### View and manage personality profiles")

                with gr.Row():
                    with gr.Column():
                        view_user_id = gr.Textbox(
                            label="User ID",
                            placeholder="Enter user ID to view profile"
                        )
                        view_btn = gr.Button("👁️ View Profile", variant="secondary")

                        profile_info = gr.Markdown(
                            value="Profile information will appear here...",
                            label="Profile Information"
                        )

                view_btn.click(
                    interface.get_profile_info,
                    inputs=[view_user_id],
                    outputs=[profile_info]
                )

        # Footer
        gr.Markdown("""
        ---
        **Instructions:**
        1. **Create Profile**: Upload training texts and optional voice sample
        2. **Chat**: Start conversation and interact with the AI personality  
        3. **Feedback**: Rate responses to improve the AI through reinforcement learning

        **Tips:**
        - Use diverse text samples for better personality modeling
        - Provide regular feedback to improve response quality
        - Clear audio samples work best for voice cloning
        """)

    return app


def main():
    """Main function to launch the Gradio interface"""
    try:
        # Create the interface
        app = create_interface()

        # Launch the app
        app.launch(
            server_name="0.0.0.0",  # Allow external access
            server_port=7860,  # Default Gradio port
            share=False,  # Set to True to create public link
            debug=True,  # Enable debug mode
            show_error=True,  # Show errors in interface
            inbrowser=True  # Open in browser automatically
        )

    except Exception as e:
        logger.error(f"Failed to launch interface: {e}")
        print(f"Error launching interface: {e}")


if __name__ == "__main__":
    main()