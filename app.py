# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, render_template

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)


# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    data_points = [0, 0, 5000, 15000, 10000, 20000, 15000, 25000, 20000, 30000, 25000, 40000]
    data_points2 = [0, 0, 5000, 22 , 10000, 20000, 15000, 25000, 20000, 30000, 25000, 40000]
    return render_template('index.html', values=data_points, values1=data_points2)



# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()
