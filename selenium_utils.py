from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

url=r'https://steelsurfer.streamlit.app'

driver.get(url)
try:
    # Find the button using the data-testid attribute
    #wake_up_button = driver.find_element_by_css_selector("[data-testid='wakeup-button-viewer']")
    wake_up_button = driver.find_element('css selector', "[data-testid='wakeup-button-viewer']")
    wake_up_button.click()
    print("App was asleep. Woke it up.")
except NoSuchElementException:
    print("App is already awake.")


# <button type="button" class="button_button__0On-O button_button_primary__E3Mmg styles_restartButton__iWmDz" data-testid="wakeup-button-viewer">Yes, get this app back up!</button>