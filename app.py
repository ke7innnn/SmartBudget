import pickle
from flask import Flask , request ,render_template,url_for
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np


with open("xgb_model.pkl","rb") as f:
    model= pickle.load(f)

with open("/Users/kevinpimenta/Desktop/projectml/scalar.pkl","rb") as f:
    ss = pickle.load(f)

label_map = {
            0: "Critical: Immediate spending review needed.",
            1: "You're on track! Try to boost savings gradually.",
            2: "Excellent! Consider investments for growth."
        }

app = Flask(__name__)
@app.route("/",methods=["GET","POST"])
def home():
    prediction = "Please enter your financial details above"
    if request.method=="POST":
        try:
            if not all(request.form.get(field) for field in ['income', 'rent', 'food', 'shopping', 'travel', 'savings']):
                    prediction = "Please fill in all fields"
            else:
                Monthly_Income= float(request.form.get("income"))
                Rent_Housing= float(request.form.get("rent"))
                Food_Groceries= float(request.form.get("food"))
                Entertainment_Shopping= float(request.form.get("shopping"))
                Travel_Transport= float(request.form.get("travel"))
                Savings_Investments= float(request.form.get("savings"))

                
                columns = [
                    "Monthly Income",
                    "Rent/Housing",
                    "Food & Groceries",
                    "Entertainment & Shopping",
                    "Travel & Transport",
                    "Savings/Investments"
                ]

                
                df = pd.DataFrame([[
                    Monthly_Income,
                    Rent_Housing,
                    Food_Groceries,
                    Entertainment_Shopping,
                    Travel_Transport,
                    Savings_Investments
                ]], columns=columns)

                df[columns] = ss.transform(df[columns])

                
                pred = model.predict(df)
                prediction = label_map[pred[0]]

        except ValueError:
            prediction = "Please enter valid numbers in all fields"
        except Exception as e:
            prediction = f"Enter Logical Number ,Error: {str(e)}"       
        
    return render_template("index.html",prediction=prediction)
        
if __name__ == "__main__":
    app.run(debug=True)




        
