def scrape_fight_matrix_data(event_url):
    """
    Scrape fight prediction data from Fight Matrix
    
    Args:
        event_url: URL to the Fight Matrix event page
    
    Returns:
        Dictionary with fight prediction data
    """
    import requests
    from bs4 import BeautifulSoup
    import re
    
    print(f"Scraping Fight Matrix data from: {event_url}")
    
    try:
        # Get the page content
        response = requests.get(event_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all fight sections
        fight_data = {}
        fight_sections = soup.find_all('p', string=re.compile(r'\*\*Last 3 Fights:'))
        
        for section in fight_sections:
            # Find fighter names
            header = section.find_previous('p')
            if not header:
                continue
                
            header_text = header.get_text()
            fighter_match = re.search(r'\[.*?\] (.*?) \(.*?\) vs\. \[.*?\] (.*?) \(', header_text)
            
            if not fighter_match:
                continue
                
            fighter1 = fighter_match.group(1)
            fighter2 = fighter_match.group(2)
            
            # Find prediction table
            table = section.find_next('table')
            if not table:
                continue
                
            rows = table.find_all('tr')[1:]  # Skip header row
            
            fight_data[(fighter1, fighter2)] = {}
            
            for row in rows:
                cells = row.find_all('td')
                if len(cells) < 4:
                    continue
                    
                system = cells[0].get_text().strip()
                favorite = cells[1].get_text().strip()
                rating = cells[2].get_text().strip()
                win_pct = cells[4].get_text().strip() if len(cells) > 4 else ""
                
                # Convert win percentage to float
                if win_pct and win_pct != "":
                    try:
                        win_pct = float(win_pct.strip('%')) / 100
                    except:
                        win_pct = 0.5
                
                # Convert betting odds to implied probability
                implied_prob = 0.5
                if system == "Betting Odds" and rating:
                    try:
                        if rating.startswith('-'):
                            # Negative odds (favorite)
                            odds = float(rating)
                            implied_prob = abs(odds) / (abs(odds) + 100)
                        else:
                            # Positive odds (underdog)
                            odds = float(rating.lstrip('+'))
                            implied_prob = 100 / (odds + 100)
                    except:
                        pass
                
                # Store the data
                system_key = system.lower().replace(' ', '_')
                fight_data[(fighter1, fighter2)][system_key] = {
                    "favorite": favorite,
                    "win_pct": win_pct if win_pct else implied_prob
                }
        
        print(f"Successfully scraped data for {len(fight_data)} fights")
        return fight_data
        
    except Exception as e:
        print(f"Error scraping Fight Matrix: {str(e)}")
        return {}

def scrape_tapology_data(event_url):
    """
    Scrape fight prediction data from Tapology
    
    Args:
        event_url: URL to the Tapology event page
    
    Returns:
        Dictionary with fight prediction data
    """
    # This would require Selenium as Tapology has more dynamic content
    # Basic implementation - expand as needed
    print(f"Scraping Tapology data from: {event_url}")
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        import time
        
        # Setup headless browser
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(event_url)
        
        # Wait for page to load
        time.sleep(3)
        
        # Extract fight data
        fight_data = {}
        fight_elements = driver.find_elements(By.CSS_SELECTOR, ".fightCard .fightCardBout")
        
        for fight_element in fight_elements:
            try:
                fighters = fight_element.find_elements(By.CSS_SELECTOR, ".fightCardFighterName")
                if len(fighters) < 2:
                    continue
                    
                fighter1 = fighters[0].text.strip()
                fighter2 = fighters[1].text.strip()
                
                # Get odds if available
                odds_elements = fight_element.find_elements(By.CSS_SELECTOR, ".odds")
                odds1 = odds_elements[0].text.strip() if len(odds_elements) > 0 else ""
                odds2 = odds_elements[1].text.strip() if len(odds_elements) > 1 else ""
                
                # Convert odds to probabilities
                prob1, prob2 = 0.5, 0.5
                
                if odds1 and odds1 != "":
                    try:
                        if odds1.startswith('-'):
                            # Negative odds (favorite)
                            odds = float(odds1)
                            prob1 = abs(odds) / (abs(odds) + 100)
                        else:
                            # Positive odds (underdog)
                            odds = float(odds1.lstrip('+'))
                            prob1 = 100 / (odds + 100)
                    except:
                        pass
                
                if odds2 and odds2 != "":
                    try:
                        if odds2.startswith('-'):
                            # Negative odds (favorite)
                            odds = float(odds2)
                            prob2 = abs(odds) / (abs(odds) + 100)
                        else:
                            # Positive odds (underdog)
                            odds = float(odds2.lstrip('+'))
                            prob2 = 100 / (odds + 100)
                    except:
                        pass
                
                # Normalize probabilities
                total = prob1 + prob2
                if total > 0:
                    prob1 = prob1 / total
                    prob2 = prob2 / total
                
                # Store the data
                fight_data[(fighter1, fighter2)] = {
                    "betting_odds": {
                        "favorite": fighter1 if prob1 > prob2 else fighter2,
                        "implied_prob": max(prob1, prob2)
                    }
                }
                
            except Exception as e:
                print(f"Error processing fight: {str(e)}")
        
        driver.quit()
        print(f"Successfully scraped data for {len(fight_data)} fights from Tapology")
        return fight_data
        
    except Exception as e:
        print(f"Error scraping Tapology: {str(e)}")
        return {}
