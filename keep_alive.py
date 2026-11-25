import os
import sys
from playwright.sync_api import sync_playwright

APP_URL = os.environ.get("APP_URL")
if not APP_URL:
    sys.exit("APP_URL 환경변수가 없습니다.")

WAKE_TEXT = "Yes, get this app back up!"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(APP_URL, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(8000)  # 페이지/앱 로딩 대기

    btn = page.get_by_text(WAKE_TEXT)
    if btn.count() > 0:
        print("앱이 잠들어 있음 -> 깨우기 버튼 클릭")
        btn.first.click()
        page.wait_for_timeout(30000)  # 앱 재기동 대기
    else:
        print("앱이 이미 켜져 있음 -> 접속 완료")

    browser.close()
