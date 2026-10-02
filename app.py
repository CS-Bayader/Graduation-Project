from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# ============================================================
# 1. تحميل النموذج والـ Scaler
# ============================================================

model = joblib.load("baseerah_model.pkl")
scaler = joblib.load("baseerah_scaler.pkl")


# ============================================================
# 2. محرك التوصيات الصحية
# ============================================================

def generate_health_recommendations(glucose, risk_code):

    diet_advice = []
    exercise_advice = []
    status_summary = ""

    # --------------------------------------------------------
    # انخفاض السكر
    # --------------------------------------------------------

    if glucose < 70:

        status_summary = (
            "Low blood glucose detected. "
            "Stop physical activity and take appropriate immediate action."
        )

        diet_advice = [
            "If you have symptoms of low blood sugar, consume a fast-acting carbohydrate such as glucose tablets or juice.",
            "Recheck your blood glucose after about 15 minutes.",
            "After recovery, consider a balanced snack containing carbohydrate and protein if another drop is possible."
        ]

        exercise_advice = [
            "Stop exercise until your blood glucose has recovered.",
            "Recheck your blood glucose before returning to physical activity.",
            "If symptoms are severe, persistent, or you are unable to safely treat yourself, seek urgent medical help."
        ]

    # --------------------------------------------------------
    # ارتفاع السكر
    # --------------------------------------------------------

    elif glucose > 180:

        status_summary = (
            "Elevated blood glucose detected. "
            "Consider monitoring your glucose and following your healthcare plan."
        )

        diet_advice = [
            "Choose meals rich in vegetables, fiber, and lean protein.",
            "Limit sugary drinks, sweets, and highly refined carbohydrates.",
            "Choose whole-grain carbohydrate sources and balanced portions.",
            "Drink water regularly unless a healthcare professional has advised you to restrict fluids."
        ]

        exercise_advice = [
            "Light to moderate physical activity such as walking may help manage blood glucose.",
            "Avoid strenuous exercise if your glucose is very high or if you have ketones.",
            "Monitor your glucose according to your usual healthcare plan."
        ]

    # --------------------------------------------------------
    # النتيجة الطبيعية
    # --------------------------------------------------------

    else:

        status_summary = (
            "Blood glucose is not in the low or elevated range "
            "based on the value entered."
        )

        diet_advice = [
            "Maintain a balanced diet containing vegetables, fiber, and lean protein.",
            "Limit excessive added sugars and highly processed foods.",
            "Choose balanced portions and regular meals."
        ]

        exercise_advice = [
            "Maintain regular physical activity according to your ability.",
            "Walking, cycling, swimming, or light resistance exercise can be options.",
            "Aim for consistent activity rather than prolonged periods of inactivity."
        ]

    return {
        "status_summary": status_summary,
        "diet": diet_advice,
        "exercise": exercise_advice
    }


# ============================================================
# 3. API الخاصة بالتنبؤ
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # استقبال البيانات من صفحة الويب
        # ----------------------------------------------------

        data = request.get_json()

        glucose = float(data["glucose"])
        age = float(data["age"])
        bmi = float(data["bmi"])
        insulin = float(data["insulin"])
        blood_pressure = float(data["blood_pressure"])

        # ----------------------------------------------------
        # التحقق من القيم
        # ----------------------------------------------------

        if glucose <= 0:
            return jsonify({
                "error": "Glucose must be greater than 0."
            }), 400

        if age <= 0:
            return jsonify({
                "error": "Age must be greater than 0."
            }), 400

        if bmi <= 0:
            return jsonify({
                "error": "BMI must be greater than 0."
            }), 400

        if insulin < 0:
            return jsonify({
                "error": "Insulin cannot be negative."
            }), 400

        if blood_pressure <= 0:
            return jsonify({
                "error": "Blood pressure must be greater than 0."
            }), 400

        # ----------------------------------------------------
        # إنشاء DataFrame بنفس ترتيب التدريب
        # ----------------------------------------------------

        input_data = pd.DataFrame([{
            "Glucose": glucose,
            "Age": age,
            "BMI": bmi,
            "Insulin": insulin,
            "BloodPressure": blood_pressure
        }])

        # ----------------------------------------------------
        # استخدام نفس الـ Scaler المستخدم أثناء التدريب
        # ----------------------------------------------------

        input_scaled = scaler.transform(input_data)

        # ----------------------------------------------------
        # التنبؤ باستخدام نموذج الذكاء الاصطناعي
        # ----------------------------------------------------

        prediction = model.predict(input_scaled)[0]

        probability = model.predict_proba(input_scaled)[0][1]

        risk_probability = round(probability * 100, 1)

        # ----------------------------------------------------
        # التوصيات
        # ----------------------------------------------------

        recommendations = generate_health_recommendations(
            glucose,
            int(prediction)
        )

        # ----------------------------------------------------
        # تحديد عنوان النتيجة
        # ----------------------------------------------------

        if prediction == 1:
            risk_title = "Higher Diabetes Risk"
        else:
            risk_title = "Lower Diabetes Risk"

        # ----------------------------------------------------
        # إرسال النتيجة إلى صفحة الويب
        # ----------------------------------------------------

        response = {
            "risk_code": int(prediction),
            "risk_probability": risk_probability,
            "risk_title": risk_title,
            "message": "Analysis completed successfully.",
            "recommendations": recommendations
        }

        return jsonify(response), 200

    except KeyError as e:

        return jsonify({
            "error": f"Missing field: {str(e)}"
        }), 400

    except ValueError:

        return jsonify({
            "error": "Please enter valid numeric values."
        }), 400

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# 4. تشغيل Flask
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=True,
        port=5000
    )
