import pyautogui
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

def get_color(s):
    if "green" in s:
        return np.array([0, 255, 0])
    elif "red" in s:
        return np.array([255, 0, 0])
    elif "blue" in s:
        return np.array([0, 0, 255])

def get_shape(s):
    if "circle" in s:
        return 5
    elif "square" in s:
        return 4
    elif "triangle" in s:
        return 3

def setup_bot():
    driver = webdriver.Chrome("D:\My Desktop\captcha\Data\chromedriver.exe")
    driver.get("file:///D:/My%20Desktop/captcha/main.html")
    begin_button = driver.find_element_by_id("start")
    begin_button.click()

    driver.switch_to_window(driver.window_handles[1])
    finish_button = driver.find_element_by_id("finishbutton")
    finish_x = finish_button.location["x"] + finish_button.size["width"]/2
    finish_y = finish_button.location["y"] + finish_button.size["height"]/4 + driver.execute_script("return window.outerHeight - window.innerHeight;")

    task = driver.find_element_by_id("task").text
    color = get_color(task)
    shape = get_shape(task)


    screen = pyautogui.screenshot()
    original = np.array(screen)
    img = np.array(screen)
    img = img[:, :, ::-1].copy() # convert to BGR
    #img = cv2.medianBlur(img, 5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
    contours, h = cv2.findContours(thresh, 1, 2)

    return contours, driver, color, shape, original, finish_x, finish_y

def add_noise(start_x, start_y, end_x, end_y, noise, n):
    x_pos = [start_x + i/n*(end_x - start_x) + random.uniform(-1*noise, noise) for i in range(1, n - 1)] + [end_x]
    y_pos = [start_y + i/n*(end_y - start_y) + random.uniform(-1*noise, noise) for i in range(1, n - 1)] + [end_y]
    return x_pos, y_pos

def bot1(travel_time, delay):
    contours, driver, color, shape, original, finish_x, finish_y = setup_bot()
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04*cv2.arcLength(cnt, True), True)
       	M = cv2.moments(cnt)
        if(M["m00"] == 0):
            continue
        x = int(M["m10"]/M["m00"])
        y = int(M["m01"]/M["m00"])
        c_shape = min(5, len(approx))
        c_color = original[y][x]
        if((c_color == color).all() and c_shape == shape):
            pyautogui.moveTo(x, y, duration=random.uniform(travel_time[0], travel_time[1]))
            pyautogui.click()
            time.sleep(random.uniform(delay[0], delay[1]))
    pyautogui.moveTo(finish_x, finish_y, duration=random.uniform(travel_time[0], travel_time[1]))
    pyautogui.click()
    time.sleep(2)
    driver.quit()


def bot2(): #instant
    contours, driver, color, shape, original, finish_x, finish_y = setup_bot()
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04*cv2.arcLength(cnt, True), True)
       	M = cv2.moments(cnt)
        if(M["m00"] == 0):
            continue
        x = int(M["m10"]/M["m00"])
        y = int(M["m01"]/M["m00"])
        c_shape = min(5, len(approx))
        c_color = original[y][x]
        if((c_color == color).all() and c_shape == shape):
            pyautogui.moveTo(x, y)
            pyautogui.click()
    pyautogui.moveTo(finish_x, finish_y)
    pyautogui.click()
    time.sleep(2)
    driver.quit()

def bot3(travel_time, delay, noise, n):
    contours, driver, color, shape, original, finish_x, finish_y = setup_bot()
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04*cv2.arcLength(cnt, True), True)
       	M = cv2.moments(cnt)
        if(M["m00"] == 0):
            continue
        x = int(M["m10"]/M["m00"])
        y = int(M["m01"]/M["m00"])
        c_shape = min(5, len(approx))
        c_color = original[y][x]
        if((c_color == color).all() and c_shape == shape):
            t = duration=random.uniform(travel_time[0], travel_time[1])
            mouse_x, mouse_y = pyautogui.position()
            x_pos, y_pos = add_noise(mouse_x, mouse_y, x, y, noise, n + 1)
            for i in range(n):
                pyautogui.moveTo(x_pos[i], y_pos[i], t/n)
            pyautogui.click()
            time.sleep(random.uniform(delay[0], delay[1]))
    pyautogui.moveTo(finish_x, finish_y, duration=random.uniform(travel_time[0], travel_time[1]))
    pyautogui.click()
    time.sleep(2)
    driver.quit()

if __name__ == "__main__":
    bot1((0.25, 0.5), (0.3, 0.5))
    #bot2()
    #bot3((0.5, 0.75), (0.3, 0.5), 100, 3)

    #for i in range(1):
    #bot3((0.5, 0.75), (0.3, 0.5), 100, 3)
    #for i in range(1):
    #    bot1((0.25, 0.5), (0.3, 0.5))

    """circles = cv2.HoughCircles(image=gray, method=cv2.HOUGH_GRADIENT, dp=0.9, minDist=80, param1=110, param2=39, maxRadius=100)
    for i in range(len(circles[0])):
        c = circles[0][i]
        x = c[0]
        y = c[1]
        r = c[2]
        print(color)
        print(x, y, end=" ")
        print(original[int(y)][int(x)], "\n")
        if(r == 10 and (color == original[int(y)][int(x)]).all()):
            pyautogui.moveTo(x, y, duration=0.5)
            pyautogui.click()"""