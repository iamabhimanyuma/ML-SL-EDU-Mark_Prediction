from flask import Flask,redirect,render_template,request
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
app=Flask(__name__)

@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
        pred_df=data.get_dataframe()
        predict_=PredictPipeline()
        result=predict_.predict(pred_df)
        print(result)
        return render_template('home.html',results=round(result[0],2))

if __name__=="__main__":
    app.run(debug=True)
