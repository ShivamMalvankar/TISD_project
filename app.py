from flask import Flask, request, jsonify, render_template
from webscarping import build_dataset
from echochamberx import analyze_dataset

app = Flask(__name__)


@app.route('/')
def home():
    # Serves the HTML file from the 'templates' folder
    return render_template('EchoChamberX.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_topic():
    try:
        data = request.json
        topic = data.get("topic")

        if not topic:
            return jsonify({"status": "error", "message": "No topic provided"}), 400

        print(f"\n[SERVER] Received request to analyze topic: '{topic}'")

        # 1. Gather the data (returns a list of dicts directly)
        print("[SERVER] Starting web scraper...")
        raw_data = build_dataset(topic, include_reddit=True)

        if not raw_data:
            return jsonify({"status": "error", "message": "Could not find enough data for this topic."}), 404

        # 2. Process the data
        print("[SERVER] Passing data to EchoChamberX NLP processor...")
        analysis_results = analyze_dataset(raw_data)

        # 3. Return results to frontend
        print("[SERVER] Analysis complete. Sending to frontend.")
        return jsonify({
            "status": "success",
            "results": analysis_results
        })

    except Exception as e:
        print(f"[SERVER ERROR] {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    print("Starting EchoChamberX Server on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)