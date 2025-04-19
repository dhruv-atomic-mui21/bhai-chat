from openai import OpenAI
from flask import Flask, request, Response, jsonify
import re

# Initialize Flask app
app = Flask(__name__)

# Initialize OpenAI clients (NVIDIA endpoint) with separate API keys
# The correct way to initialize OpenAI client with base_url and api_key
text_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="text_api_key"  # Replace with your actual text API key
)

code_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="code_api_key"  # Replace with your actual code API key
)

# Helper function to stream completions as SSE with appropriate client
def stream_completion(client, system_prompt, user_message):
    try:
        completion = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            stream=True
        )
        for chunk in completion:
            content = chunk.choices[0].delta.get("content", "")
            if content:
                yield f"data: {content}\n\n"
    except Exception as e:
        yield f"data: [ERROR] {str(e)}\n\n"

# Route for general text generation
@app.route('/generate-text', methods=['POST'])
def generate_text():
    data = request.get_json()
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "Message content required"}), 400
    return Response(
        stream_completion(text_client, "You are a helpful assistant.", user_message),
        content_type='text/event-stream'
    )

# Route for code generation
@app.route('/generate-code', methods=['POST'])
def generate_code():
    data = request.get_json()
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "Message content required"}), 400
    return Response(
        stream_completion(code_client, "You are a helpful coding senior developer. Generate clean and well-structured code.", user_message),
        content_type='text/event-stream'
    )

# ========== TEXT POLISHING FUNCTIONS ==========

@app.route('/polish-text', methods=['POST'])
def polish_text_endpoint():
    """
    POST JSON: { "text": "<text to polish>", "polish_level": "low|medium|high", "style": "formal|creative|technical" }
    Returns polished text with improved grammar, formatting, and style.
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify(error="`text` field is required"), 400

    text = data["text"]
    polish_level = data.get("polish_level", "medium")  # low, medium, high
    style = data.get("style", "formal")  # formal, creative, technical

    try:
        polished_text = polish_text(text, polish_level, style)
        return jsonify({
            'status': 'success',
            'message': 'Text polished successfully',
            'original_text': text,
            'polished_text': polished_text,
            'params': {
                'polish_level': polish_level,
                'style': style
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error polishing text: {str(e)}'
        }), 500

def polish_text(text, level='medium', style='formal'):
    """
    Polish text using basic string operations and AI for higher-level polishing
    Level can be 'low', 'medium', or 'high'
    Style can be 'formal', 'creative', 'technical'
    """
    if not text:
        return text
    
    # Basic cleanup for all levels
    # Fix common capitalization issues
    text = fix_capitalization(text)
    
    # Fix common punctuation issues
    text = fix_punctuation(text)
    
    # Fix common spacing issues
    text = fix_spacing(text)
    
    if level == 'low':
        # Just basic corrections
        return text
    
    # Medium: Add simple filtering and readability improvements
    if level == 'medium' or level == 'high':
        # Filter common profanity
        text = filter_profanity(text)
        
        # Fix redundant words
        text = remove_word_redundancies(text)
        
        # Fix common grammar issues
        text = fix_common_grammar(text)
    
    # High: Use AI to enhance the text
    if level == 'high':
        try:
            # Use text_client for text operations
            system_prompt = f"""
            You are a professional text editor specializing in {style} writing.
            Polish the text to make it more {style} while maintaining its original meaning and intent.
            Fix any grammatical errors, improve word choice, and enhance the overall flow.
            Return ONLY the polished text without any explanations or comments.
            """
            
            response = text_client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.2,
                max_tokens=1024
            )
            
            ai_polished_text = response.choices[0].message.content
            
            # Verify the AI didn't return an empty or much shorter text
            if ai_polished_text and len(ai_polished_text) >= len(text) * 0.7:
                return ai_polished_text
        except Exception as e:
            print(f"Error in AI text polishing: {str(e)}")
            # Continue with basic polishing if AI fails
    
    return text

def fix_capitalization(text):
    """Fix basic capitalization issues"""
    # Capitalize first letter of sentences
    sentences = re.split(r'([.!?]\s+)', text)
    result = ""
    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        if i < len(sentences) - 1:
            ending = sentences[i+1]
        else:
            ending = ""
        
        if sentence and sentence[0].isalpha():
            sentence = sentence[0].upper() + sentence[1:]
        
        result += sentence + ending
    
    # Capitalize 'I' when it's a word
    result = re.sub(r'\bi\b', 'I', result)
    
    return result

def fix_punctuation(text):
    """Fix common punctuation issues"""
    # Fix spaces before punctuation
    text = re.sub(r'\s+([.,;:!?)])', r'\1', text)
    
    # Fix spaces after opening brackets
    text = re.sub(r'([([{])\s+', r'\1', text)
    
    # Ensure proper spacing after punctuation
    text = re.sub(r'([.,;:!?])\s*(\w)', lambda m: m.group(1) + ' ' + m.group(2), text)
    
    # Fix multiple punctuation
    text = re.sub(r'\.{2,}', '...', text)  # Convert any number of dots to ellipsis
    text = re.sub(r'!{2,}', '!', text)  # Remove multiple exclamation marks
    text = re.sub(r'\?{2,}', '?', text)  # Remove multiple question marks
    
    return text

def fix_spacing(text):
    """Fix spacing issues"""
    # Remove excess spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove spaces at beginning and end
    text = text.strip()
    
    return text

def filter_profanity(text):
    """Simple function to filter profanity"""
    # Very basic list of words to filter
    profane_words = ['damn', 'hell', 'crap', 'ass', 'shit', 'fuck', 'bitch']
    
    # Replace profane words with asterisks
    for word in profane_words:
        pattern = re.compile(r'\b' + word + r'\b', re.IGNORECASE)
        replacement = '*' * len(word)
        text = pattern.sub(replacement, text)
    
    return text

def remove_word_redundancies(text):
    """Remove redundant words and phrases"""
    # List of redundant phrases and their replacements
    redundancies = [
        (r'\b(absolutely|completely|totally|utterly) essential\b', 'essential'),
        (r'\b(basic|fundamental) essentials\b', 'essentials'),
        (r'\bcurrent status\b', 'status'),
        (r'\badvance planning\b', 'planning'),
        (r'\bfuture plans\b', 'plans'),
        (r'\bpast history\b', 'history'),
        (r'\bvery unique\b', 'unique'),
        (r'\band etc\b', 'etc.'),
        (r'\bfree gift\b', 'gift'),
        (r'\bnew innovation\b', 'innovation'),
        (r'\brepeat again\b', 'repeat'),
        (r'\brevert back\b', 'revert'),
        (r'\breturn back\b', 'return')
    ]
    
    for pattern, replacement in redundancies:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def fix_common_grammar(text):
    """Fix common grammatical errors"""
    # Common grammar mistakes and their corrections
    grammar_fixes = [
        # Subject-verb agreement
        (r'\b(I|you|we|they) (was)\b', r'\1 were'),
        (r'\b(he|she|it|one) (were)\b', r'\1 was'),
        # Common confusions
        (r'\btheir (is|are)\b', r'there \1'),
        (r'\byour (is|are|has|have)\b', r"you're \1"),
        (r'\bits (is|has)\b', r"it's \1"),
        # Missing articles
        (r'\b(go to|at) ([a-z]+ity\b)', r'\1 the \2'),
        (r'\b(go to|at) ([a-z]+versity\b)', r'\1 the \2'),
        # Incorrect prepositions
        (r'\bdifferent (to|than)\b', 'different from'),
        (r'\bin regards (of|to)\b', 'regarding'),
        # Double negatives
        (r'\bdon\'t (have|know) (no|nothing)\b', r'don\'t \1 anything'),
        (r'\bcan\'t (have|find|see) (no|nothing)\b', r'can\'t \1 anything'),
        # Comma splices
        (r'([^,])(\s+)(however|nevertheless|therefore|thus|consequently)(\s+)', r'\1, \3\4'),
    ]
    
    # Apply grammar fixes
    for pattern, replacement in grammar_fixes:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

# ========== CODE POLISHING FUNCTIONS ==========

@app.route('/polish-code', methods=['POST'])
def polish_code_endpoint():
    """
    POST JSON: { "code": "<code to polish>", "language": "<programming language>", "polish_level": "low|medium|high" }
    Returns polished code with improved formatting and style.
    """
    data = request.get_json(silent=True)
    if not data or "code" not in data:
        return jsonify(error="`code` field is required"), 400

    code = data["code"]
    language = data.get("language", "python")
    polish_level = data.get("polish_level", "medium")  # low, medium, high

    try:
        polished_code = polish_code(code, language, polish_level)
        return jsonify({
            'status': 'success',
            'message': 'Code polished successfully',
            'original_code': code,
            'polished_code': polished_code,
            'params': {
                'language': language,
                'polish_level': polish_level
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error polishing code: {str(e)}'
        }), 500

def polish_code(code, language="python", level="medium"):
    """
    Polish code to improve formatting, style, and best practices.
    Uses AI-based polishing for high-quality results.
    
    Parameters:
    - code: The source code to polish
    - language: Programming language of the code
    - level: 'low', 'medium', or 'high' polishing intensity
    
    Returns:
    - Polished code as a string
    """
    if not code:
        return code
        
    # For lower polish levels, just do basic formatting
    if level == "low":
        return format_code_basic(code, language)
        
    # For medium and high levels, use AI polishing
    system_message = f"""
    You are an expert {language} developer specializing in code quality and best practices.
    
    Polish the provided {language} code with these guidelines:
    - Fix indentation and consistent formatting
    - Add or improve comments for clarity
    - Rename variables for better readability
    - Apply {language} best practices and conventions
    - Fix potential bugs or edge cases
    - {'Optimize performance where possible' if level == 'high' else ''}
    - {'Improve error handling' if level == 'high' else ''}
    
    Return ONLY the polished code without explanations.
    """
    
    try:
        # Use the code_client for code operations
        response = code_client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": code}
            ],
            temperature=0.1,  # Keep temperature low for consistency
            max_tokens=1024
        )
        
        polished_code = response.choices[0].message.content
        
        # Check if response is empty or significantly shorter than original
        if not polished_code or len(polished_code) < len(code) * 0.5:
            return format_code_basic(code, language)  # Fallback to basic formatting
            
        return polished_code
        
    except Exception as e:
        # Fallback to basic formatting on errors
        print(f"Error in AI code polishing: {str(e)}")
        return format_code_basic(code, language)

def format_code_basic(code, language):
    """
    Basic code formatting without relying on external libraries.
    Handles basic indentation and spacing issues.
    """
    lines = code.split('\n')
    formatted_lines = []
    
    # Language-specific formatting
    if language.lower() in ["python", "py"]:
        indent_level = 0
        for line in lines:
            # Handle indentation
            stripped = line.strip()
            
            # Check for indentation decrease
            if stripped.startswith(('return', 'break', 'continue', 'pass', 'else:', 'elif', 'except:', 'finally:')):
                if indent_level > 0:
                    indent_level -= 1
                    
            # Add proper indentation
            formatted_line = '    ' * indent_level + stripped
            formatted_lines.append(formatted_line)
            
            # Check for indentation increase
            if stripped.endswith((':', '{', '{')):
                indent_level += 1
                
            # Check for block end
            if stripped in ['}', 'endif', 'endwhile', 'endfor']:
                if indent_level > 0:
                    indent_level -= 1
                    
    elif language.lower() in ["javascript", "js", "typescript", "ts"]:
        # Handle JavaScript/TypeScript
        brace_count = 0
        for line in lines:
            stripped = line.strip()
            
            # Handle braces for indentation
            open_braces = stripped.count('{')
            close_braces = stripped.count('}')
            
            # Adjust for closing braces at start of line
            if stripped.startswith('}'):
                brace_count -= 1
                
            # Add proper indentation
            formatted_line = '  ' * brace_count + stripped
            formatted_lines.append(formatted_line)
            
            # Adjust the brace count for next line
            brace_count = brace_count + open_braces - close_braces
            
    else:
        # Generic formatting for other languages
        formatted_lines = [line.strip() for line in lines]
    
    return '\n'.join(formatted_lines)

if __name__ == '__main__':
    app.run(debug=True)