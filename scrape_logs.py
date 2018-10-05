from scrapy.selector import Selector
from selenium import webdriver
import pickle
import time

build_fail_flag = "//div/div[@class='summary-item'][1]/div/div[@class='badge-label']"
xpath_filter = "//span[@class='pre'][1]/span[@class='white'][1]"
url = 'https://circleci.com/gh/palantir/gradle-docker/'
expand_header = "//div[@class='build-output']/div/div[@class='ah_wrapper']/div/div/span[@class='command-text']"
success_contents = "//div[@class='button success contents']"
failure_contents = "//div[@class='button failed contents']"

data = []
driver = webdriver.Chrome('/Users/timlee/webdrivers/chromedriver')


for i in range(81,201):
    url_idx = str(290 - i)
    driver.get(url + url_idx)
    time.sleep(1)
    
    build_status = driver.find_element_by_xpath(build_fail_flag)
    build_status = build_status.get_attribute("innerHTML")

    # open headers
    success_headers = driver.find_elements_by_xpath(success_contents)
    for head in success_headers:
        print(head.get_attribute('class'))
        head.click()
        time.sleep(1)

    failure_headers = driver.find_elements_by_xpath(failure_contents)
    for head in failure_headers:
        print(head.get_attribute('class'))
        head.click()
        time.sleep(1)



    build_log = driver.find_elements_by_xpath(xpath_filter)
    build_log = [log.get_attribute('innerHTML') for log in build_log]
    print(i, build_status)

    data.append({
        'build_status' : build_status,
        'build_log': build_log
    })
    if (i % 20 == 0) & (i > 0):
        with open('log_data_%s.pkl' % str(i).zfill(3), 'wb') as f:
            pickle.dump(data, f)
        data = []
driver.close()
