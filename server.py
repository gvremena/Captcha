from aiohttp import web
import os
import data_loader
import logistic_regression
import socketio
import numpy as np

mu = 1
sigma = 2
p = 3

theta = np.load("theta.npy");

sio = socketio.AsyncServer(cors_allowed_origins="*")
app = web.Application()
sio.attach(app)

async def index(request):
    with open("captcha.html") as f:
        return web.Response(text=f.read(), content_type="text/html")

@sio.on("message")
async def print_message(sid, message):
    print("Socket ID: " , sid)
    print(message)

@sio.on("load_data")
async def load_data(sid, json):
    print("Adding new json!")
    if not os.path.exists(data_loader.data_path):
        np.save("data.npy", np.empty((0, 6)))

    data = data_loader.load_data(json)

    result = logistic_regression.predict_p(data, theta) #data_loader.predict(data, mu, sigma, p)
    print(data)
    print("result: ", result)
    await sio.emit("message", str(result[0]))

app.router.add_get("/", index)

if __name__ == "__main__":
    mu = np.load("mu.npy")
    sigma = np.load("sigma.npy")
    p = np.load("p.npy")
    web.run_app(app)
    print(theta)