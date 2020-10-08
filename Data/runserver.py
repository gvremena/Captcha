from aiohttp import web
import os
import sys
import data_loader
import logistic_regression
import socketio
import numpy as np

theta = np.load("theta.npy")
prt = os.getenv('PORT', 32123)

sio = socketio.AsyncServer(cors_allowed_origins="*")
app = web.Application()
sio.attach(app)

async def index(request):
    prt = os.getenv('PORT', 32123)
    with open("captcha.html") as f:
        page = f.read().replace("29968", str(prt))
        return web.Response(text=page, content_type="text/html")
    sys.stderr.write("INDEX OPENING\n")
    sys.stderr.flush()

@sio.on("message")
async def print_message(sid, message):
    print("Socket ID: " , sid)
    print(message)

@sio.on("load_data")
async def load_data(sid, json):
    #print("Adding new json!")
    sys.stderr.write("SERVER RECEIVING\n")
    sys.stderr.flush()
    
    if not os.path.exists(data_loader.data_path):
        np.save("data.npy", np.empty((0, 6)))

    data = data_loader.load_data(json)

    result = logistic_regression.predict_p(data, theta) #data_loader.predict(data, mu, sigma, p)
    print(data)
    print("result: ", result)
    await sio.emit("message", str(result[0]))

app.router.add_get("/", index)

if __name__ == "__main__":
    #web.run_app(app)
    sys.stderr.write("PRINT TEST\n")
    sys.stderr.flush()
    web.run_app(app, port=prt)
