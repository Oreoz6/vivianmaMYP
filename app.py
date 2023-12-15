#import libraries 
import getpass 
from flask import Flask, render_template, request, url_for, jsonify, send_file 
import requests
import numpy as np
# import jinja2
# import tempfile
# import uuid 
import pickle 
from matplotlib import pyplot as plt 

# get processing helper function to process input to predicting features 
def input_to_feature(data): 
    gen = data[0].lower() 
    age = int(data[1]) 
    we = int(data[2])
    sen = data[3].lower() 
    ind = data[4].lower() 
    inc = int(data[5]) 
    inv = int(data[6]) 
    edu = data[-1].lower() 

    # process gender 
    temp = 0 
    if "Female" in gen: temp += 1
    elif "Male" in gen: temp -= -1
    if "Transgender" in gen: temp /= 2
    if "Non" in gen: temp *= 2 
    gen = temp 
    
    # process age 
    temp = 0 
    if age < 18: temp = 1 
    elif age < 25: temp = 2 
    elif age < 35: temp = 3
    elif age < 45: temp = 4
    elif age < 55: temp = 5
    elif age < 65: temp = 6
    else: temp = 7
    age = temp 

    # process work experience  
    temp = 0 
    if we < 3: temp = 1 
    elif we < 6: temp = 2 
    elif we < 11: temp = 3
    elif we < 16: temp = 4
    else: we = 5 
    we = temp 
    
    # process seniority 
    temp = 0 
    if "Not" in sen or "Other" in sen: temp = -1 
    elif "Self" in sen: temp = 0
    elif "Intern" in sen: temp = 1
    elif "Associate" in sen: temp = 2
    elif "Manager" in sen: temp = 3
    elif "Executive" in sen: temp = 4 
    else: temp = 5 
    sen = temp 

    # process industry  
    temp = 0 
    if "Not" in ind or "Other" in ind: temp = -1
    elif 'Agriculture' in ind or "Food" in ind or 'Oil' in ind or 'Construction' in ind or 'Non-profit' in ind: temp = 1
    elif "Education" in ind or "Academia" in ind: temp = 2
    elif "News" in ind or "Healthcare" in ind: temp = 3
    elif "Energy" in ind or "Technology" in ind or "Government" in ind or "Real" in ind: temp = 4
    elif "Retired" in ind: temp = 5 
    else: temp = 6 
    ind = temp 

    # process income   
    temp = 0 
    if inc < 32100: temp = 1 
    elif inc < 50001: temp = 2 
    elif inc < 70001: temp = 3
    elif inc < 100001: temp = 4
    elif inc < 200001: temp = 5
    else: temp = 6 
    inc = temp 

    # process current invested    
    temp = 0 
    if inv < 10: temp = 1 
    elif inv < 26: temp = 2 
    elif inv < 50: temp = 3
    elif inv < 75: temp = 4 
    else: temp = 5 
    inv = temp 

    # process education 
    temp = 0 
    if "Other" in edu: temp = -1 
    elif "Some" in edu: temp = 1
    elif "diploma" in edu: temp = 2
    elif "University" in edu: temp = 3
    elif "Master" in edu: temp = 4
    else: temp = 5 
    edu = temp 
    
    return [[gen, age, we, sen, ind, inc, inv, edu]] 

#Initialize the flask App and load model
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# global variables 
gen_op = ["Cis Female", "Cis Male", "Transgender Female", "Transgender Male", "Non-binary", "Other"] 
sen_op = ["Not applicable", "Self-employed", "Intern", "Associate", "Manager", "Executive", "Owner / Partner"]
ind_op = ["Agriculture", "Oil & Gas", "Construction & Manufacturing", "Education", "Non-profit", "Academia", "News & Media", "Healthcare & Pharmaceuticals", "Energy", "Computer & Information Technology", "Government", "Retired", "Banking & Finance", "Food & Drinks", "Real Estate", "Other"] 
ind_op = sorted(ind_op) 
edu_op = ["Some high school", "High shool diploma", "University or College", "Master's", "Doctorate (Ph.D., M.D., J.D., etc.)", "Other"] 

# Get dictionary of labels vs descriptors 
letters = {} 
letters['1H'] = "  You have a relatively large risk appetite for managing your investments. You prefer to take control and full oversight over your own investments rather than putting them in the hands of an agent. You are less interested in investments that present less potential for rapid and significant fluctuations, and are more interested in high-risk high-return options such as single-name stocks or alternatives. Your investment experience is characterized by thrill and constant monitoring of performance." 
letters['1L'] = "  You have a relatively small risk appetite for managing your investments. You prefer to let professional agents or asset managers handle or influence your investments rather than taking complete control and oversight. You are less interested in high-risk high-return investments, and are more interested in stable options that impose low risk but are also less likely to increase in value rapidly, such as government bonds or index funds. Playing it safe and being cautious is one characteristic of your investment-ality."
letters['2R'] = "  Moreover, you have a strong focus on the financial materiality of your investments rather than their philanthropy considerations. For exampe, the oil and gas industries may not necessarily be labeled 'green', but the cpaital gains may be very attractive to you."
letters['2E'] = "  Moreover, you take the environmental and social implications of your investments into consideraion, and do not simply focus on the financial gains. For exampe, the cpaital gains in the oil and gas industries may be very attractive, but these industries are not environmentally sustainable and its supply chain could present human rights violations. Such considerations for environmental and community wellbeing may make the option less attractive to your investment decision-making." 
letters['3L'] = "  You favor investments that are relatively long-term, such as real-estate, government bonds, or mutual funds. These investments may not generate immediate profit, and may require large input, but do have less transaction fees and act as assets that acts as a safety-net against short-term financial disturbances."
letters['3S'] = "  You favor foreseeable investment returns that occur in the near future, without having to wait 5, 10, or 20 years to collect your gians. These are very attractive investment options as they offer financial flexibility and the input to any one investment is typically lower compared to longer term investments."
letters['4D'] = "  Lastly, you tend to rely more on financial information sources that provide quantitative evidence or analysis to help make their investment decisions. Examples include reading company financial reports, getting data from professional advisors, or analyzing stock market trends. Essentially, you are less likely to jump onto an investment hype train without obtaining sufficient data as evidence of promising performance."
letters['4Q'] = "  Lastly, you tend to rely more on qualitative sources for information that may impact their investment decisions. Some examples include consulting friend and peers for their opinions, observing investment trends on social media, inferring the performance of an asset from relevant information in the use, and so forth. You might be susceptible to jump onto an invetment hype train such as NFTs or Crypto early on due to the general widespread enthusiasm."




# get index page written in HTML (home)
@app.route('/') 
def home(): 
    return render_template('index.html', gen_options = gen_op, sen_options = sen_op, ind_options = ind_op, edu_options = edu_op) 

# implement result page 
@app.route('/get_inputs', methods=['GET', 'POST'])
def get_inputs():
    temp = request.args.getlist("inputs") 
    final_features = input_to_feature(temp) 
    prediction = model.predict(final_features)[0] 

    j = 1 
    explanation = []
    for i in list(prediction): 
        temp = str(j) + i 
        explanation.append(letters[temp])
        j += 1 

    return render_template('results.html', text=prediction, p1=explanation[0], p2=explanation[1], p3=explanation[2], p4=explanation[3]) 

# implement thank you page 
@app.route('/ty', methods=['POST'])
def ty():
    fb = request.form['feedback']
    fb_txt = request.form['text_feedback'] 
    if fb: 
        f = open("feedback_btn.txt", "a")
        f.write(str(fb)+"\n") 
        f.close()
    if fb_txt: 
        f = open("feedback_text.txt", "a")
        f.write(str(fb_txt)+"\n") 
        f.close()

    return render_template('thankyou.html') 

# implement gallery page 
@app.route('/gallery')
def gallery(): 
    return render_template('gallery.html', brainstorm_src = "static\Brainstorm.png", fb_src = "static\channel1.png", reddit_src = "static\channel2.png", twitter_src = "static\channel3.png", photo_src = "static\Vivian.png",
                           gen_all_src = "static\Gen_all.png", gen_sd_src = "static\Gen_sd.png", gen_di_src = "static\Gen_di.png",
                           ind_all_src = "static\Ind_all.png", ind_sd_src = "static\Ind_sd.png", ind_di_src = "static\Ind_di.png", 
                           currinv_src = "static\CurrInv.png", esg_src = "static\ESG.png", source_src="static\Source.png",
                           time_src = "static\Timestamp.png", k_src = "static\K.png", err_src = "static\Err.png") 

# start server
if __name__ == "__main__":
    app.run(port=5000, debug=True) 