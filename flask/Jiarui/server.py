
"""
Columbia's COMS W4111.001 Introduction to Databases
Example Webserver
To run locally:
    python server.py
Go to http://localhost:8111 in your browser.
A debugger such as "pdb" may be helpful for debugging.
Read about it online.
"""
import os
  # accessible as a variable in index.html:
from sqlalchemy import *
from sqlalchemy.pool import NullPool
from flask import Flask, request, render_template, g, redirect, Response

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)


#
# The following is a dummy URI that does not connect to a valid database. You will need to modify it to connect to your Part 2 database in order to use the data.
#
# XXX: The URI should be in the format of: 
#
#     postgresql://USER:PASSWORD@35.243.220.243/proj1part2
#
# For example, if you had username gravano and password foobar, then the following line would be:
#
#     DATABASEURI = "postgresql://gravano:foobar@35.243.220.243/proj1part2"
#
DATABASEURI = "postgresql://sl4653:6906@35.231.103.173/proj1part2"


#
# This line creates a database engine that knows how to connect to the URI above.
#
engine = create_engine(DATABASEURI)


@app.before_request
def before_request():
  """
  This function is run at the beginning of every web request 
  (every time you enter an address in the web browser).
  We use it to setup a database connection that can be used throughout the request.

  The variable g is globally accessible.
  """
  try:
    g.conn = engine.connect()
  except:
    print("uh oh, problem connecting to database")
    import traceback; traceback.print_exc()
    g.conn = None

@app.teardown_request
def teardown_request(exception):
  """
  At the end of the web request, this makes sure to close the database connection.
  If you don't, the database could run out of memory!
  """
  try:
    g.conn.close()
  except Exception as e:
    pass


@app.route('/')
def index():
  print(request.args)
 

  return render_template("index.html")




@app.route('/recipes')
def recipes():
  
  print(request.args)
  cursor = g.conn.execute("SELECT recipe_name FROM recipe")
  names = []
  for result in cursor:
    names.append(result['recipe_name'])  # can also be accessed using result[0]
  cursor.close()
  context = dict(data = names)
  return render_template("recipes.html", **context)



@app.route('/search', methods=['GET'])
def search():
  print(request.args)
  name = request.args.get('name')
  name = '%' + name + '%'
  print(name)

  cursor = g.conn.execute('SELECT recipe_name FROM recipe WHERE recipe_name LIKE (%s)', name)
  recipes = []
  for result in cursor:
    recipes.append(result['recipe_name']) 
  cursor.close()

  cursor = g.conn.execute('SELECT recipe_name FROM recipe, ingredient WHERE ingredient.recipe_id = recipe.recipe_id AND ingredient_name LIKE (%s)', name)
  for result in cursor:
    recipes.append(result['recipe_name']) 
  cursor.close()

  recipes = list(dict.fromkeys(recipes))


  context = dict(data = recipes)
  return render_template('search.html', **context)

@app.route('/toprating', methods=['GET'])
def toprating():
  print(request.args)
  cursor = g.conn.execute("SELECT distinct recipe_name, AVG(review.rating) as avg FROM recipe, review WHERE review.recipe_id = recipe.recipe_id GROUP BY recipe_name ORDER BY avg DESC")
  names = []
  for result in cursor:
    names.append(result['recipe_name'])  # can also be accessed using result[0]
  cursor.close()
  context = dict(data = names)
  return render_template("recipes.html", **context)


@app.route('/recipe', methods=['GET'])
def recipe():
  print(request.args)

  name = request.args.get('name')
  recipename = name

  cursor = g.conn.execute('SELECT descriptions FROM recipe WHERE recipe_name = (%s)', name)
  descriptions = []
  for result in cursor:
    descriptions.append(result['descriptions']) 
  cursor.close()

  cursor = g.conn.execute('SELECT steps FROM recipe WHERE recipe_name = (%s)', name)
  steps = []
  for result in cursor:
    steps.append(result['steps']) 
  cursor.close()

  cursor = g.conn.execute('SELECT minute FROM recipe WHERE recipe_name = (%s)', name)
  minutes = []
  for result in cursor:
    minutes.append(result['minute']) 
  cursor.close()

  cursor = g.conn.execute('SELECT tag_content FROM recipe,subtag,tag WHERE recipe.recipe_id=subtag.recipe_id AND subtag.tag_id=tag.tag_id AND recipe_name = (%s)', name)
  tags = []
  for result in cursor:
    tags.append(result['tag_content']) 
  cursor.close()


  cursor = g.conn.execute('SELECT review_content FROM recipe, review WHERE review.recipe_id = recipe.recipe_id AND recipe_name = (%s)', name)
  reviews = []
  for result in cursor:
    reviews.append(result['review_content']) 
  cursor.close()

  cursor = g.conn.execute('SELECT rating FROM recipe, review WHERE review.recipe_id = recipe.recipe_id AND recipe_name = (%s)', name)
  ratings = []
  for result in cursor:
    ratings.append(result['rating']) 
  cursor.close()

  total = 0

  for n in ratings:
    total += n
  if len(ratings) > 0:
    ratings = round(total/len(ratings), 1)
  else:
    ratings.append("There is no rating for this recipe yet.")

  if len(reviews) == 0:
    reviews.append("There is no review for this recipe yet.")
  context = dict(description = descriptions, step = steps, tag=tags, minute=minutes, review = reviews, rating = ratings, recipename = recipename)
  return render_template('recipe.html', **context)





if __name__ == "__main__":
  import click

  @click.command()
  @click.option('--debug', is_flag=True)
  @click.option('--threaded', is_flag=True)
  @click.argument('HOST', default='127.0.0.1')
  @click.argument('PORT', default=8111, type=int)
  def run(debug, threaded, host, port):
    """
    This function handles command line parameters.
    Run the server using:

        python server.py

    Show the help text using:

        python server.py --help

    """

    HOST, PORT = host, port
    print("running on %s:%d" % (HOST, PORT))
    app.run(host=HOST, port=PORT, debug=True, threaded=threaded)

  run()

