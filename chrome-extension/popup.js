// UFC Data Collector - Popup Script

document.addEventListener('DOMContentLoaded', async () => {
  const collectBtn = document.getElementById('collect-btn');
  const sendBtn = document.getElementById('send-btn');
  const copyBtn = document.getElementById('copy-btn');
  const clearBtn = document.getElementById('clear-btn');
  const statusEl = document.getElementById('status');
  const countEl = document.getElementById('fighter-count');
  const collectedDataEl = document.getElementById('collected-data');
  const fighterListEl = document.getElementById('fighter-list');

  // Load stored data
  const stored = await chrome.storage.local.get(['collectedFighters', 'sources']);
  let collectedFighters = stored.collectedFighters || {};
  let sources = stored.sources || { tapology: false, bfo: false, fightmatrix: false };

  updateUI();

  // Collect data from current page
  collectBtn.addEventListener('click', async () => {
    try {
      statusEl.className = 'status collecting';
      statusEl.textContent = 'Collecting data...';

      // Get current tab
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      const url = tab.url;

      // Determine which scraper to use based on URL
      let scraperFn;
      let sourceName;

      if (url.includes('tapology.com')) {
        scraperFn = scrapeTapology;
        sourceName = 'tapology';
      } else if (url.includes('bestfightodds.com')) {
        scraperFn = scrapeBestFightOdds;
        sourceName = 'bfo';
      } else if (url.includes('fightmatrix.com')) {
        scraperFn = scrapeFightMatrix;
        sourceName = 'fightmatrix';
      } else {
        throw new Error('Not on a supported site (Tapology, BestFightOdds, or FightMatrix)');
      }

      // Inject and execute the scraper
      const results = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: scraperFn
      });

      if (results && results[0] && results[0].result) {
        const data = results[0].result;
        console.log('Scraped data:', data);

        if (data.fighters.length === 0) {
          statusEl.className = 'status error';
          statusEl.textContent = `No fighters found on this page. Check console for debug info.`;
          return;
        }

        // Merge new data with existing
        for (const fighter of data.fighters) {
          const existingKeys = Object.keys(collectedFighters);
          const key = findMatchingKey(fighter.name, existingKeys);
          if (!collectedFighters[key]) {
            collectedFighters[key] = { name: fighter.name };
          }
          // Merge data sources
          if (fighter.tapology !== undefined) {
            // Store tapology as nested object with consensus and method breakdown
            // This matches the structure expected by prediction-engine.js
            if (typeof collectedFighters[key].tapology !== 'object') {
              collectedFighters[key].tapology = {};
            }
            collectedFighters[key].tapology.consensus = fighter.tapology;
            sources.tapology = true;
          }
          if (fighter.bfoWinPct !== undefined) {
            if (!collectedFighters[key].bfo) collectedFighters[key].bfo = {};
            collectedFighters[key].bfo.winPct = fighter.bfoWinPct;
            sources.bfo = true;
          }
          if (fighter.bfoMethodKO !== undefined) {
            if (!collectedFighters[key].bfo) collectedFighters[key].bfo = {};
            collectedFighters[key].bfo.methodKO = fighter.bfoMethodKO;
            collectedFighters[key].bfo.methodSub = fighter.bfoMethodSub;
            collectedFighters[key].bfo.methodDec = fighter.bfoMethodDec;
          }
          if (fighter.cirrs !== undefined) {
            collectedFighters[key].cirrs = fighter.cirrs;
            sources.fightmatrix = true;
          }
          // Merge expanded FightMatrix data
          if (fighter.eloK170 !== undefined) {
            if (!collectedFighters[key].fightmatrix) collectedFighters[key].fightmatrix = {};
            collectedFighters[key].fightmatrix.eloK170 = fighter.eloK170;
          }
          if (fighter.eloMod !== undefined) {
            if (!collectedFighters[key].fightmatrix) collectedFighters[key].fightmatrix = {};
            collectedFighters[key].fightmatrix.eloMod = fighter.eloMod;
          }
          if (fighter.glicko !== undefined) {
            if (!collectedFighters[key].fightmatrix) collectedFighters[key].fightmatrix = {};
            collectedFighters[key].fightmatrix.glicko = fighter.glicko;
          }
          if (fighter.whr !== undefined) {
            if (!collectedFighters[key].fightmatrix) collectedFighters[key].fightmatrix = {};
            collectedFighters[key].fightmatrix.whr = fighter.whr;
          }
          if (fighter.bettingWinPct !== undefined) {
            if (!collectedFighters[key].fightmatrix) collectedFighters[key].fightmatrix = {};
            collectedFighters[key].fightmatrix.bettingWinPct = fighter.bettingWinPct;
            collectedFighters[key].fightmatrix.bettingOdds = fighter.bettingOdds;
          }
          if (fighter.age !== undefined) {
            if (!collectedFighters[key].fightmatrix) collectedFighters[key].fightmatrix = {};
            collectedFighters[key].fightmatrix.age = fighter.age;
          }
          if (fighter.daysSinceLastFight !== undefined) {
            if (!collectedFighters[key].fightmatrix) collectedFighters[key].fightmatrix = {};
            collectedFighters[key].fightmatrix.daysSinceLastFight = fighter.daysSinceLastFight;
          }
          if (fighter.ranking !== undefined) {
            if (!collectedFighters[key].fightmatrix) collectedFighters[key].fightmatrix = {};
            collectedFighters[key].fightmatrix.ranking = fighter.ranking;
          }
          if (fighter.record !== undefined) {
            if (!collectedFighters[key].fightmatrix) collectedFighters[key].fightmatrix = {};
            collectedFighters[key].fightmatrix.record = fighter.record;
          }
          if (fighter.last3Record !== undefined) {
            if (!collectedFighters[key].fightmatrix) collectedFighters[key].fightmatrix = {};
            collectedFighters[key].fightmatrix.last3Record = fighter.last3Record;
          }
          // Merge method prediction data (TKO/SUB/DEC) into tapology object
          if (fighter.tko !== undefined || fighter.sub !== undefined || fighter.dec !== undefined) {
            if (typeof collectedFighters[key].tapology !== 'object') {
              collectedFighters[key].tapology = {};
            }
            if (fighter.tko !== undefined) collectedFighters[key].tapology.koTko = fighter.tko;
            if (fighter.sub !== undefined) collectedFighters[key].tapology.sub = fighter.sub;
            if (fighter.dec !== undefined) collectedFighters[key].tapology.dec = fighter.dec;
          }
        }

        // Save to storage
        await chrome.storage.local.set({ collectedFighters, sources });

        statusEl.className = 'status ready';
        statusEl.textContent = `Collected ${data.fighters.length} fighters from ${data.source}`;
        updateUI();
      } else {
        throw new Error('No data returned from scraper');
      }
    } catch (error) {
      console.error('Collection error:', error);
      statusEl.className = 'status error';
      statusEl.textContent = 'Error: ' + error.message;
    }
  });

  // Copy to clipboard
  copyBtn.addEventListener('click', async () => {
    const fighters = Object.values(collectedFighters);
    const json = JSON.stringify(fighters, null, 2);
    await navigator.clipboard.writeText(json);
    statusEl.textContent = 'Copied to clipboard!';
    setTimeout(() => {
      statusEl.textContent = 'Ready to collect data';
    }, 2000);
  });

  // Clear data
  clearBtn.addEventListener('click', async () => {
    collectedFighters = {};
    sources = { tapology: false, bfo: false, fightmatrix: false };
    await chrome.storage.local.set({ collectedFighters, sources });
    updateUI();
    statusEl.textContent = 'Data cleared';
  });

  // Send to App
  sendBtn.addEventListener('click', async () => {
    try {
      const fighters = Object.values(collectedFighters);
      if (fighters.length === 0) return;

      statusEl.className = 'status collecting';
      statusEl.textContent = 'Sending to app...';

      // Send to localhost
      const response = await fetch('http://localhost:5555/api/import', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ fighters: fighters, source: 'extension' })
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const result = await response.json();
      console.log('Server response:', result);

      statusEl.className = 'status ready';
      statusEl.textContent = `Sent ${fighters.length} fighters to App!`;

      // Flash success
      sendBtn.textContent = 'Sent Successfully!';
      setTimeout(() => {
        sendBtn.textContent = 'Send to App (Localhost)';
        statusEl.textContent = 'Ready to collect data';
      }, 3000);

    } catch (error) {
      console.error('Send error:', error);
      statusEl.className = 'status error';
      statusEl.textContent = 'Failed to send: ' + error.message + '. Is the app running?';
    }
  });

  // Collect results (post-event)
  const collectResultsBtn = document.getElementById('collect-results-btn');
  collectResultsBtn.addEventListener('click', async () => {
    try {
      statusEl.className = 'status collecting';
      statusEl.textContent = 'Collecting results...';

      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      const url = tab.url;

      let scraperFn;
      if (url.includes('tapology.com')) {
        scraperFn = scrapeTapologyResults;
      } else if (url.includes('ufc.com')) {
        scraperFn = scrapeUFCResults;
      } else {
        throw new Error('Results scraping only supported on Tapology or UFC.com event pages');
      }

      const results = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: scraperFn
      });

      const data = results[0].result;
      if (!data || !data.results || data.results.length === 0) {
        throw new Error('No results found on this page');
      }

      // Copy results to clipboard
      const json = JSON.stringify(data.results, null, 2);
      await navigator.clipboard.writeText(json);

      statusEl.className = 'status ready';
      statusEl.textContent = `Copied ${data.results.length} fight results to clipboard!`;
    } catch (error) {
      statusEl.className = 'status error';
      statusEl.textContent = error.message;
    }
  });

  function updateUI() {
    const fighterArray = Object.values(collectedFighters);
    countEl.textContent = fighterArray.length;
    copyBtn.disabled = fighterArray.length === 0;
    if (sendBtn) sendBtn.disabled = fighterArray.length === 0;

    // Update source badges
    document.getElementById('badge-tapology').className = 'badge' + (sources.tapology ? ' active' : '');
    document.getElementById('badge-bfo').className = 'badge' + (sources.bfo ? ' active' : '');
    document.getElementById('badge-fightmatrix').className = 'badge' + (sources.fightmatrix ? ' active' : '');

    // Show/hide collected data
    if (fighterArray.length > 0) {
      collectedDataEl.style.display = 'block';
      fighterListEl.innerHTML = fighterArray.map(f => {
        // Handle nested tapology structure
        const tapologyConsensus = typeof f.tapology === 'object' ? f.tapology.consensus : f.tapology;
        return `
        <div class="fighter-item">
          <span class="fighter-name">${f.name}</span>
          <span class="fighter-data">
            ${tapologyConsensus !== undefined ? `T:${tapologyConsensus}%` : ''}
            ${f.bfo?.winPct !== undefined ? `BFO:${f.bfo.winPct}%` : ''}
            ${f.cirrs !== undefined ? `FM:${f.cirrs}` : ''}
          </span>
        </div>
      `;
      }).join('');
    } else {
      collectedDataEl.style.display = 'none';
    }
  }

  function normalizeName(name) {
    // Remove accents (Natália -> Natalia)
    const withoutAccents = name.normalize('NFD').replace(/[\u0300-\u036f]/g, '');
    // Lowercase
    const lower = withoutAccents.toLowerCase();
    // Expand common abbreviations (St. -> Saint, Jr. -> Junior, etc.)
    const expanded = lower
      .replace(/\bst\.\s*/g, 'saint ')
      .replace(/\bjr\.\s*/g, 'junior ')
      .replace(/\bsr\.\s*/g, 'senior ');
    // Remove non-alpha chars and collapse whitespace
    const cleaned = expanded.replace(/[^a-z\s]/g, '').replace(/\s+/g, ' ').trim();
    // Sort name parts alphabetically to handle "Song Yadong" vs "Yadong Song"
    const parts = cleaned.split(' ').sort();
    return parts.join(' ');
  }

  // Find existing key that matches a fighter name (handles partial matches)
  // Known aliases: FightMatrix legal names -> common ring names
  const FIGHTER_ALIASES = {
    'renato carneiro': 'renato moicano',
    'renato moicano': 'renato carneiro',
    'song yadong': 'yadong song',
    'yadong song': 'song yadong',
  };

  function findMatchingKey(name, existingKeys) {
    const normalizedNew = normalizeName(name);

    // Exact match first
    if (existingKeys.includes(normalizedNew)) {
      return normalizedNew;
    }

    // Check alias table
    const withoutAccents = name.normalize('NFD').replace(/[\u0300-\u036f]/g, '');
    const cleaned = withoutAccents.toLowerCase().replace(/[^a-z\s]/g, '').replace(/\s+/g, ' ').trim();
    const alias = FIGHTER_ALIASES[cleaned];
    if (alias) {
      const normalizedAlias = normalizeName(alias);
      if (existingKeys.includes(normalizedAlias)) {
        return normalizedAlias;
      }
    }

    // Get last name (last word after normalization, before sorting)
    const nameParts = cleaned.split(' ');
    const lastName = nameParts[nameParts.length - 1];
    const firstName = nameParts[0];

    // Look for partial matches by last name + first name
    for (const key of existingKeys) {
      const keyParts = key.split(' ');
      // Check if last name and first name both appear in the key
      if (keyParts.includes(lastName) && keyParts.includes(firstName)) {
        return key;
      }
      // Check if it's just first + last vs full name (e.g., "ateba gautier" vs "abega ateba gautier")
      if (keyParts.includes(lastName) && keyParts.some(p => p === firstName || firstName.includes(p) || p.includes(firstName))) {
        return key;
      }
    }

    // No match found, return normalized new name
    return normalizedNew;
  }
});

// ============================================================
// SCRAPER FUNCTIONS - These run in the context of the web page
// ============================================================

function scrapeTapology() {
  console.log('[UFC Scraper] Running Tapology scraper...');
  const fighters = [];
  const seenNames = new Set();

  // Words to filter out - these aren't fighter names
  const filterWords = ['decision', 'submission', 'knockout', 'events', 'bouts', 'round', 'method', 'main', 'prelim', 'card', 'fight', 'view', 'save', 'pick', 'help', 'left', 'right', 'choose', 'previous', 'next', 'ranking', 'record', 'unranked', 'event', 'co-main', 'preliminary'];

  const fullText = document.body.innerText;
  console.log('[UFC Scraper] Page text sample:', fullText.substring(0, 500));

  // Try to extract method bars from DOM
  // Tapology uses classes like: tko_bar_slim, sub_bar_slim, dec_bar_slim (or total_bar_slim)
  // The width style contains the percentage (e.g., style="width: 35.5%")
  const methodDataByName = {};

  // Helper to extract width percentage from element
  const getWidthPct = (el) => {
    if (!el) return 0;
    const width = el.style?.width || '';
    const match = width.match(/([\d.]+)%/);
    return match ? Math.round(parseFloat(match[1])) : 0;
  };

  // Strategy 1: Look for fighter containers with method bars inside
  // Tapology often has fighter info in sections/divs with links to fighter pages
  const fighterLinks = document.querySelectorAll('a[href*="/fightcenter/fighters/"]');
  console.log('[UFC Scraper] Found fighter links:', fighterLinks.length);

  fighterLinks.forEach((link, idx) => {
    const name = link.textContent.trim();
    if (!name || name.length < 3 || filterWords.includes(name.toLowerCase())) return;

    // Look for method bars in the parent containers (go up a few levels)
    let container = link.parentElement;
    for (let i = 0; i < 5 && container; i++) {
      // Look for bar elements with _slim suffix (Tapology's actual class names)
      const tkoBar = container.querySelector('[class*="tko_bar"], .tko_bar_slim');
      const subBar = container.querySelector('[class*="sub_bar"], .sub_bar_slim');
      const decBar = container.querySelector('[class*="dec_bar"], .dec_bar_slim');

      if (tkoBar || subBar || decBar) {
        const key = name.toLowerCase();
        if (!methodDataByName[key]) {
          methodDataByName[key] = { name };
        }

        const tko = getWidthPct(tkoBar);
        const sub = getWidthPct(subBar);
        const dec = getWidthPct(decBar);

        if (tko > 0) methodDataByName[key].tko = tko;
        if (sub > 0) methodDataByName[key].sub = sub;
        if (dec > 0) methodDataByName[key].dec = dec;

        console.log('[UFC Scraper] Found method bars for', name, '- TKO:', tko, 'SUB:', sub, 'DEC:', dec);
        break; // Found bars, stop climbing
      }
      container = container.parentElement;
    }
  });

  // Strategy 1b: Extract pro MMA records from DOM
  // Supports two page layouts:
  //   A) Event page: [data-bout-wrapper] divs with title='UFC Record: X-X' tooltips + <span> pro record
  //   B) Predictions page: div[id^="boutNum"] with table.fighterStats, 2nd row has "Pro Record At Fight"
  const recordByName = {};

  // --- Layout A: Event page (bout wrappers with tooltips) ---
  const boutWrappers = document.querySelectorAll('[data-bout-wrapper]');
  console.log('[UFC Scraper] Found event page bout wrappers:', boutWrappers.length);

  boutWrappers.forEach(wrapper => {
    const links = wrapper.querySelectorAll('a[href*="/fightcenter/fighters/"]');
    const seenInBout = new Set();

    links.forEach(link => {
      const name = link.textContent.trim();
      if (!name || name.length < 3 || seenInBout.has(name.toLowerCase())) return;
      seenInBout.add(name.toLowerCase());

      let bioContainer = link.closest('[id*="Bio"]') || link.closest('[id*="boutFullsize"]');
      if (!bioContainer) return;

      const recordDiv = bioContainer.querySelector('[title*="UFC Record"]');
      if (recordDiv) {
        const recordSpan = recordDiv.querySelector('span');
        if (recordSpan) {
          let record = recordSpan.textContent.trim();
          if (record.match(/^\d+-\d+$/) && !record.match(/^\d+-\d+-\d+$/)) {
            record = record + '-0';
          }
          if (record.match(/^\d+-\d+-\d+$/)) {
            recordByName[name.toLowerCase()] = record;
            console.log('[UFC Scraper] [EventPage] Found pro record for', name, ':', record);
          }
        }
      }
    });
  });

  // --- Layout B: Predictions page (boutNum sections with textContent parsing) ---
  // The fighterStats tables exist in DOM but are visually collapsed.
  // Using textContent (not innerText) captures hidden content too.
  const boutSections = document.querySelectorAll('div[id^="boutNum"]');
  console.log('[UFC Scraper] Found predictions page bout sections:', boutSections.length);

  boutSections.forEach(section => {
    // Use textContent to get ALL text including hidden/collapsed content
    const sectionText = section.textContent;

    // Look for "Pro Record At Fight" pattern with records on either side
    // textContent layout: "...7-0-0\n...Pro Record At Fight\n...13-7-0..."
    const proRecordIdx = sectionText.indexOf('Pro Record');
    if (proRecordIdx === -1) return;

    // Extract text around the "Pro Record" label and find W-L-D patterns
    const before = sectionText.substring(Math.max(0, proRecordIdx - 200), proRecordIdx);
    const after = sectionText.substring(proRecordIdx, Math.min(sectionText.length, proRecordIdx + 200));

    // Find records: last W-L-D before label = fighter1, first W-L-D after label = fighter2
    const beforeRecords = before.match(/\d+-\d+-\d+/g);
    const afterRecords = after.match(/\d+-\d+-\d+/g);

    const record1 = beforeRecords ? beforeRecords[beforeRecords.length - 1] : null;
    const record2 = afterRecords ? afterRecords[afterRecords.length - 1] : null;

    if (!record1 && !record2) return;

    // Get fighter names from links (deduplicate by last name)
    const fighterLinks = section.querySelectorAll('a[href*="/fightcenter/fighters/"]');
    const names = [];
    fighterLinks.forEach(link => {
      const name = link.textContent.trim();
      if (!name || name.length < 3 || filterWords.includes(name.toLowerCase())) return;
      const keyLastName = name.split(' ').pop().toLowerCase();
      const existingIdx = names.findIndex(n => {
        const nLower = n.toLowerCase();
        const nLastName = n.split(' ').pop().toLowerCase();
        return nLower.includes(name.toLowerCase()) || name.toLowerCase().includes(nLower) ||
               (nLastName === keyLastName && nLastName.length >= 3);
      });
      if (existingIdx >= 0) {
        if (name.length > names[existingIdx].length) names[existingIdx] = name;
      } else if (names.length < 2) {
        names.push(name);
      }
    });

    console.log('[UFC Scraper] [PredsPage] Bout names:', names, 'records:', record1, '/', record2);

    if (record1 && names[0] && !recordByName[names[0].toLowerCase()]) {
      recordByName[names[0].toLowerCase()] = record1;
      console.log('[UFC Scraper] [PredsPage] Record for', names[0], ':', record1);
    }
    if (record2 && names[1] && !recordByName[names[1].toLowerCase()]) {
      recordByName[names[1].toLowerCase()] = record2;
      console.log('[UFC Scraper] [PredsPage] Record for', names[1], ':', record2);
    }
  });

  // --- Layout C: Text-based record extraction from innerText ---
  // Works on event pages where records appear near fighter names as:
  //   "FighterName\nmanage_search\n7-0\n#31" (event page)
  // Also works on predictions pages if "Pro Record At Fight" text is visible:
  //   "7-0-0\nPro Record At Fight\n13-7-0"
  console.log('[UFC Scraper] Attempting text-based record extraction...');

  const textLines = fullText.split('\n').map(l => l.trim()).filter(l => l.length > 0);
  for (let i = 0; i < textLines.length; i++) {
    const line = textLines[i];

    // Pattern: "Pro Record At Fight" label between two records
    if (line.includes('Pro Record') && line.includes('At Fight')) {
      // Look for records above and below this line
      const recAbove = (textLines[i - 1] || '').trim();
      const recBelow = (textLines[i + 1] || '').trim();
      // Also check 2 lines above/below in case of spacing
      const recAbove2 = (textLines[i - 2] || '').trim();
      const recBelow2 = (textLines[i + 2] || '').trim();

      const rec1 = recAbove.match(/^\d+-\d+-\d+$/) ? recAbove : (recAbove2.match(/^\d+-\d+-\d+$/) ? recAbove2 : null);
      const rec2 = recBelow.match(/^\d+-\d+-\d+$/) ? recBelow : (recBelow2.match(/^\d+-\d+-\d+$/) ? recBelow2 : null);

      if (rec1 || rec2) {
        // Find the nearest "vs" matchup above this line to identify fighters
        for (let j = i - 1; j >= Math.max(0, i - 30); j--) {
          const vsLine = textLines[j].replace(/\s+(Main Card|Prelim|Early Prelim)\b.*$/i, '').replace(/\s*\|.*$/, '');
          const vsMatch = vsLine.match(/^(.+?)\s+vs\.?\s+(.+?)(?:\s+[IVX]+)?$/i);
          if (vsMatch) {
            const f1 = vsMatch[1].trim();
            const f2 = vsMatch[2].trim();
            if (rec1 && !recordByName[f1.toLowerCase()]) {
              recordByName[f1.toLowerCase()] = rec1;
              console.log('[UFC Scraper] [Text] Found record for', f1, ':', rec1);
            }
            if (rec2 && !recordByName[f2.toLowerCase()]) {
              recordByName[f2.toLowerCase()] = rec2;
              console.log('[UFC Scraper] [Text] Found record for', f2, ':', rec2);
            }
            break;
          }
        }
      }
    }

    // Pattern: event page - record appears after "manage_search" icon text
    // "FighterName" -> "manage_search" -> "7-0" or "7-0-0"
    if (line === 'manage_search') {
      const nextLine = (textLines[i + 1] || '').trim();
      let record = nextLine;
      if (record.match(/^\d+-\d+$/) && !record.match(/^\d+-\d+-\d+$/)) {
        record = record + '-0';
      }
      if (record.match(/^\d+-\d+-\d+$/)) {
        // Look backward for a fighter name (skip manage_search, find last text that's a name)
        for (let j = i - 1; j >= Math.max(0, i - 5); j--) {
          const prevLine = textLines[j].trim();
          if (prevLine.length >= 3 && !filterWords.includes(prevLine.toLowerCase()) &&
              !prevLine.match(/^\d/) && !prevLine.includes('manage_search') &&
              prevLine.match(/^[A-Za-zÀ-ÿ\-'.\s]+$/)) {
            // This could be a name or part of a name - find the full name
            // Look for fighter links to get the canonical name
            const key = prevLine.toLowerCase();
            if (!recordByName[key]) {
              recordByName[key] = record;
              console.log('[UFC Scraper] [Text/EventPage] Found record for', prevLine, ':', record);
            }
            break;
          }
        }
      }
    }
  }

  // --- Layout D: Broader bout wrapper search ---
  // If Layout A missed fighters (no [title*="UFC Record"] tooltip), try extracting W-L-D patterns
  // from the bout wrapper's full text near fighter name links
  if (boutWrappers.length > 0) {
    boutWrappers.forEach(wrapper => {
      const links = wrapper.querySelectorAll('a[href*="/fightcenter/fighters/"]');
      const seenInBout = new Set();

      links.forEach(link => {
        const name = link.textContent.trim();
        if (!name || name.length < 3 || seenInBout.has(name.toLowerCase())) return;
        if (recordByName[name.toLowerCase()]) return; // Already have record from earlier strategy
        seenInBout.add(name.toLowerCase());

        // Search nearby text for W-L-D pattern (within the bout wrapper)
        const wrapperText = wrapper.textContent || '';
        // Find the name in the text and look for a record nearby
        const nameIdx = wrapperText.toLowerCase().indexOf(name.toLowerCase());
        if (nameIdx === -1) return;

        // Look for W-L-D pattern within 200 chars after the name
        const nearbyText = wrapperText.substring(nameIdx, nameIdx + 200);
        const recMatch = nearbyText.match(/\b(\d{1,2}-\d{1,2}(?:-\d{1,2})?)\b/);
        if (recMatch) {
          let record = recMatch[1];
          if (record.match(/^\d+-\d+$/) && !record.match(/^\d+-\d+-\d+$/)) {
            record = record + '-0';
          }
          if (record.match(/^\d+-\d+-\d+$/)) {
            recordByName[name.toLowerCase()] = record;
            console.log('[UFC Scraper] [Layout D] Found record for', name, ':', record);
          }
        }
      });
    });
  }

  console.log('[UFC Scraper] Total records extracted:', Object.keys(recordByName).length, recordByName);

  // Helper to normalize names for matching (strips apostrophes, hyphens, accents, etc.)
  const normalizeForMatch = (name) => {
    // First normalize accents (Natália -> Natalia)
    const withoutAccents = name.normalize('NFD').replace(/[\u0300-\u036f]/g, '');
    // Then lowercase and remove non-alpha chars
    return withoutAccents.toLowerCase().replace(/[^a-z\s]/g, '').replace(/\s+/g, ' ').trim();
  };

  // Strategy 2: Collect method bars by index
  // IMPORTANT: Method bars are ordered by FAVORITE (higher consensus %), not by left/right position
  // Bar 0,2,4... = favorite's method breakdown for each fight
  // Bar 1,3,5... = underdog's method breakdown for each fight
  // We store bars by index now, then match to fighters AFTER we know consensus percentages
  console.log('[UFC Scraper] Collecting method bars by index...');

  // Helper to extract width percentage
  const extractWidth = (el) => {
    if (!el) return 0;
    const width = el.style?.width || '';
    const match = width.match(/([\d.]+)%/);
    return match ? Math.round(parseFloat(match[1])) : 0;
  };

  // Get all method bars in DOM order
  const allTkoBars = Array.from(document.querySelectorAll('[class*="tko_bar"]'));
  const allSubBars = Array.from(document.querySelectorAll('[class*="sub_bar"]'));
  const allDecBars = Array.from(document.querySelectorAll('[class*="dec_bar"]'));
  console.log('[UFC Scraper] Found bars - TKO:', allTkoBars.length, 'SUB:', allSubBars.length, 'DEC:', allDecBars.length);

  // Store method data by bar index (will match to fighters later based on consensus %)
  const methodBarsByIndex = [];
  for (let i = 0; i < allTkoBars.length; i++) {
    const tko = extractWidth(allTkoBars[i]);
    const sub = extractWidth(allSubBars[i]);
    const dec = extractWidth(allDecBars[i]);
    methodBarsByIndex.push({ tko, sub, dec });
    console.log('[UFC Scraper] Bar', i, '- TKO:', tko, 'SUB:', sub, 'DEC:', dec);
  }

  console.log('[UFC Scraper] Stored', methodBarsByIndex.length, 'method bar sets for later matching');

  // Build fight matchups list from "Fighter1 vs Fighter2" patterns
  // These appear in fight card order on the page (main event first)
  const fightMatchups = []; // Array of {fighter1: name, fighter2: name}
  // Context-aware line parser to handle duplicate names (e.g., Javid Basharat vs Farid Basharat)
  const consensusByName = {};

  // --- STATE MACHINE APPROACH ---

  // Normalize diacritics for matching: ł→l, é→e, etc.
  // NFD decomposition handles most (é, ñ, ü, etc.) but not ł, ø, đ which are single codepoints
  const stripDiacritics = (text) => {
    return text
      .normalize('NFD')
      .replace(/[\u0300-\u036f]/g, '')
      .replace(/[łŁ]/g, m => m === 'ł' ? 'l' : 'L')
      .replace(/[øØ]/g, m => m === 'ø' ? 'o' : 'O')
      .replace(/[đĐ]/g, m => m === 'đ' ? 'd' : 'D');
  };

const newLines = fullText.split('\n').map(l => l.trim()).filter(l => l.length > 0);
let activeFighters = [];
let lastFighterSeen = null;

for (let i = 0; i < newLines.length; i++) {
  const line = stripDiacritics(newLines[i]);

  // Detect Matchup Header
  // Strip trailing card info (e.g. "Main Card | Middleweight · 185 lbs | ...")
  const vsLine = line.replace(/\s+(Main Card|Prelim|Early Prelim)\b.*$/i, '').replace(/\s*\|.*$/, '');
  const vsMatch = vsLine.match(/^([A-Z][a-zA-Z\-'.]+(?:\s+[A-Z][a-zA-Z\-'.]+)*)\s+vs\.?\s+([A-Z][a-zA-Z\-'.]+(?:\s+[A-Z][a-zA-Z\-'.]+)*)(?:\s+[IVX]+)?$/i);
  if (vsMatch) {
    const f1 = vsMatch[1].trim();
    const f2 = vsMatch[2].trim().replace(/\s+[IVX]+$/, '');
    // Validate
    const isName = (n) => !filterWords.includes(n.split(' ').pop().toLowerCase()) && n.split(' ').length <= 4;
    if (isName(f1) && isName(f2)) {
      activeFighters = [f1, f2];
      fightMatchups.push({ fighter1: f1, fighter2: f2 });

      // Init entries
      const k1 = f1.toLowerCase();
      const k2 = f2.toLowerCase();
      if (!consensusByName[k1]) consensusByName[k1] = { name: f1 };
      if (!consensusByName[k2]) consensusByName[k2] = { name: f2 };

      console.log('[UFC Scraper] Context:', f1, 'vs', f2);
    }
    continue;
  }

  if (activeFighters.length === 2) {
    // We are inside a matchup block

    // Check for "Name 55%" (Single line) — handles both "Oleksiejczuk 95%" and "J. Oleksiejczuk 95%"
    const combinedMatch = line.match(/^((?:[A-Z]\.?\s+)?[A-Z][a-zA-Z\-'.]+)\s+(\d{1,3})%$/);
    if (combinedMatch) {
      const name = combinedMatch[1].trim();
      const pct = parseInt(combinedMatch[2]);
      // Match to active fighters using normalized last name comparison
      const nameLower = name.toLowerCase();
      const target = activeFighters.find(f => {
        const fLower = f.toLowerCase();
        const fLastName = f.split(' ').pop().toLowerCase();
        return fLower.includes(nameLower) || fLastName === nameLower || nameLower.includes(fLastName);
      });
      if (target) {
        const key = target.toLowerCase();
        consensusByName[key].tapology = pct;
        console.log(`[UFC Scraper] Matched combined "${name}" -> ${target} (${pct}%)`);
      }
      continue;
    }

    // Check for "55%" (Multi-line)
    const pctMatch = line.match(/^(\d{1,3})%$/);
    if (pctMatch) {
      const pct = parseInt(pctMatch[1]);
      if (lastFighterSeen && pct > 0 && pct <= 100) {
        const key = lastFighterSeen.toLowerCase();
        consensusByName[key].tapology = pct;
        lastFighterSeen = null; // Reset
      }
      continue;
    }

    // This line might be just a Name?
    // Check if it matches one of our active fighters
    const matchedF = activeFighters.find(f => {
      const parts = f.split(' ');
      const lastName = parts[parts.length - 1];
      return line.includes(lastName) || line === f || (line.length > 3 && f.includes(line));
    });

    if (matchedF && !filterWords.includes(line.toLowerCase())) {
      lastFighterSeen = matchedF;
    }
  }
}

console.log('[UFC Scraper] Found', fightMatchups.length, 'fight matchups');
console.log('[UFC Scraper] Found consensus for', Object.keys(consensusByName).length, 'fighters');

// Now process fights in order, matching bars based on favorite/underdog
// Bars are in fight card order: fight 0 bars at 0,1; fight 1 bars at 2,3; etc.
console.log('[UFC Scraper] Matching method bars to fighters by fight order and consensus %...');

for (let fightIdx = 0; fightIdx < fightMatchups.length; fightIdx++) {
  const matchup = fightMatchups[fightIdx];
  const f1Key = matchup.fighter1.toLowerCase();
  const f2Key = matchup.fighter2.toLowerCase();

  const f1Data = consensusByName[f1Key];
  const f2Data = consensusByName[f2Key];

  if (!f1Data || !f2Data) {
    console.log('[UFC Scraper] Fight', fightIdx, '- Missing consensus data for', matchup.fighter1, 'or', matchup.fighter2);
    continue;
  }

  const barIdx1 = fightIdx * 2;     // First bar for this fight (favorite's bar)
  const barIdx2 = fightIdx * 2 + 1; // Second bar for this fight (underdog's bar)

  if (barIdx1 >= methodBarsByIndex.length || barIdx2 >= methodBarsByIndex.length) {
    console.log('[UFC Scraper] Fight', fightIdx, '- No bars available at indices', barIdx1, barIdx2);
    continue;
  }

  const favoriteBar = methodBarsByIndex[barIdx1];
  const underdogBar = methodBarsByIndex[barIdx2];

  // Determine who is favorite (higher consensus %)
  const f1IsFavorite = f1Data.tapology >= f2Data.tapology;

  if (f1IsFavorite) {
    // Fighter1 is favorite, gets bar at even index
    if (favoriteBar.tko > 0) f1Data.tko = favoriteBar.tko;
    if (favoriteBar.sub > 0) f1Data.sub = favoriteBar.sub;
    if (favoriteBar.dec > 0) f1Data.dec = favoriteBar.dec;
    // Fighter2 is underdog, gets bar at odd index
    if (underdogBar.tko > 0) f2Data.tko = underdogBar.tko;
    if (underdogBar.sub > 0) f2Data.sub = underdogBar.sub;
    if (underdogBar.dec > 0) f2Data.dec = underdogBar.dec;
  } else {
    // Fighter2 is favorite, gets bar at even index
    if (favoriteBar.tko > 0) f2Data.tko = favoriteBar.tko;
    if (favoriteBar.sub > 0) f2Data.sub = favoriteBar.sub;
    if (favoriteBar.dec > 0) f2Data.dec = favoriteBar.dec;
    // Fighter1 is underdog, gets bar at odd index
    if (underdogBar.tko > 0) f1Data.tko = underdogBar.tko;
    if (underdogBar.sub > 0) f1Data.sub = underdogBar.sub;
    if (underdogBar.dec > 0) f1Data.dec = underdogBar.dec;
  }
}

  // Merge DOM-extracted records into fighter data
  // Uses multiple matching strategies: exact, normalized, last name, partial
  for (const [key, data] of Object.entries(consensusByName)) {
    // Try exact match first
    let record = recordByName[key] || recordByName[normalizeForMatch(data.name)];

    // Try last name match (handles event page "manage_search" pattern which often has just last name)
    if (!record) {
      const lastName = data.name.split(' ').pop().toLowerCase();
      if (lastName.length >= 3 && recordByName[lastName]) {
        record = recordByName[lastName];
      }
    }

    // Try partial match: any recordByName key that's a substring of the fighter name or vice versa
    if (!record) {
      const nameLower = data.name.toLowerCase();
      for (const [recKey, recVal] of Object.entries(recordByName)) {
        if (recKey.length >= 3 && (nameLower.includes(recKey) || recKey.includes(nameLower))) {
          record = recVal;
          break;
        }
      }
    }

    if (record) {
      data.record = record;
      console.log('[UFC Scraper] Merged record for', data.name, ':', record);
    }
  }

  // Re-build clean fighters array
  const finalFighters = Object.values(consensusByName);

  console.log('[UFC Scraper] Final fighters with method data:', finalFighters);
  return { source: 'Tapology', fighters: finalFighters };
}

async function scrapeBestFightOdds() {
  console.log('[UFC Scraper] Running BestFightOdds scraper...');
  const fighters = [];

  // Step 0: Expand all prop sections so method-of-victory odds load into the DOM
  // Each fight has a td.prop-cell-exp with data-mu="matchupId" — clicking it expands props
  const expandBtns = document.querySelectorAll('td.prop-cell-exp');
  const expandedMus = new Set();
  let expandCount = 0;
  for (const btn of expandBtns) {
    const mu = btn.getAttribute('data-mu');
    if (mu && !expandedMus.has(mu)) {
      expandedMus.add(mu);
      btn.click();
      expandCount++;
    }
  }
  if (expandCount > 0) {
    console.log(`[UFC Scraper] Expanded props for ${expandCount} matchups, waiting for DOM...`);
    await new Promise(resolve => setTimeout(resolve, 1500)); // wait for prop rows to load
  }

  // BFO DOM structure:
  // Two parallel tables: odds-table-responsive-header (names) + odds-table in table-scroller (odds)
  // Matchup rows: <tr id="mu-XXXXX"> for Fighter A, next <tr> for Fighter B
  // Prop rows: <tr class="pr"> with text like "Fighter wins by TKO/KO"
  // Odds cells: <td class="but-sg" data-li="[bookId,pos,matchupId]"><span>-150</span></td>

  // Helper: convert American odds to implied probability (0-100)
  function oddsToProb(odds) {
    if (odds < 0) return Math.abs(odds) / (Math.abs(odds) + 100) * 100;
    return 100 / (odds + 100) * 100;
  }

  // Build set of column indices that are real sportsbooks (skip Polymarket/Kalshi)
  // Uses data-b attribute on th elements: data-b="28" = Polymarket, data-b="29" = Kalshi
  function getSportsbookColumns(table) {
    const validColumns = new Set();
    const predictionMarkets = ['polymarket', 'kalshi'];
    const headers = table.querySelectorAll('thead th');
    headers.forEach((th, idx) => {
      const bookName = (th.textContent || '').toLowerCase().trim();
      const isPredictionMarket = predictionMarkets.some(pm => bookName.includes(pm));
      const isPropsHeader = th.classList.contains('table-prop-header');
      if (!isPredictionMarket && !isPropsHeader && idx > 0) {
        validColumns.add(idx);
      }
    });
    console.log('[UFC Scraper] Sportsbook columns:', [...validColumns].join(','),
      'Skipped:', [...headers].filter((th, idx) => !validColumns.has(idx) && idx > 0).map(th => th.textContent.trim().split('\n')[0]).join(', '));
    return validColumns;
  }

  // Helper: get consensus odds from a row's cells (median of real sportsbook odds)
  // Skips prediction market columns (Polymarket, Kalshi) which use different pricing
  function getConsensusOdds(row, validColumns) {
    const cells = row.querySelectorAll('td');
    const odds = [];
    cells.forEach((cell, idx) => {
      // Each row has <th> (name) then <td> cells, so td[0] = column 1 in the table.
      // validColumns uses thead th indices (0-based including the name th),
      // so offset td index by +1 to align with thead indices.
      const colIdx = idx + 1;
      if (validColumns && validColumns.size > 0 && !validColumns.has(colIdx)) return;
      // Skip prop-cell and button-cell elements
      if (cell.classList.contains('prop-cell') || cell.classList.contains('button-cell')) return;
      const span = cell.querySelector('span');
      if (!span) return;
      const text = span.textContent.replace(/[▼▲]/g, '').trim();
      const num = parseInt(text, 10);
      if (!isNaN(num) && text.match(/^[+-]?\d+$/)) {
        odds.push(num);
      }
    });
    if (odds.length === 0) return null;
    // Return median odds for consensus
    odds.sort((a, b) => a - b);
    return odds[Math.floor(odds.length / 2)];
  }

  // Find the scrollable odds table (has actual odds, not just names)
  const scrollerDiv = document.querySelector('.table-scroller');
  if (!scrollerDiv) {
    console.log('[UFC Scraper] No .table-scroller found, trying direct table');
  }
  const oddsTable = scrollerDiv
    ? scrollerDiv.querySelector('table.odds-table')
    : document.querySelector('table.odds-table:not(.odds-table-responsive-header)');

  // Find the responsive header table (has fighter names)
  const nameTable = document.querySelector('table.odds-table-responsive-header');

  if (!nameTable && !oddsTable) {
    console.log('[UFC Scraper] No odds tables found');
    return { source: 'BestFightOdds', fighters: [] };
  }

  // Use the name table for fighter names + prop labels, odds table for odds values
  const sourceTable = nameTable || oddsTable;
  const rows = sourceTable.querySelectorAll('tbody tr');
  const oddsRows = oddsTable ? oddsTable.querySelectorAll('tbody tr') : rows;

  console.log('[UFC Scraper] Found', rows.length, 'rows in name table');

  // Build valid sportsbook column set (skip Polymarket/Kalshi)
  const validColumns = oddsTable ? getSportsbookColumns(oddsTable) : new Set();
  console.log('[UFC Scraper] Valid sportsbook columns:', validColumns.size);

  // Parse matchups: collect pairs of fighters by matchup ID
  const matchups = {}; // matchupId -> { fighterA: {name, odds}, fighterB: {name, odds}, props: {} }
  let currentMatchupId = null;
  let currentFighterIdx = 0;

  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const oddsRow = i < oddsRows.length ? oddsRows[i] : row;

    // Check if this is a matchup row (fighter row, not prop)
    if (row.classList.contains('pr')) {
      // This is a prop row — extract method props
      if (!currentMatchupId || !matchups[currentMatchupId]) continue;
      const propText = row.querySelector('th')?.textContent?.trim() || '';
      const propOdds = getConsensusOdds(oddsRow, validColumns);

      if (propOdds !== null) {
        const mu = matchups[currentMatchupId];
        // Match prop patterns: "[Name] wins by TKO/KO", "[Name] wins by submission", "[Name] wins by decision"
        const koMatch = propText.match(/^(.+?)\s+wins by TKO\/KO$/i);
        const subMatch = propText.match(/^(.+?)\s+wins by submission$/i);
        const decMatch = propText.match(/^(.+?)\s+wins by decision$/i);

        if (koMatch) {
          const propName = koMatch[1].trim();
          if (!mu.props[propName]) mu.props[propName] = {};
          mu.props[propName].ko = propOdds;
        } else if (subMatch) {
          const propName = subMatch[1].trim();
          if (!mu.props[propName]) mu.props[propName] = {};
          mu.props[propName].sub = propOdds;
        } else if (decMatch) {
          const propName = decMatch[1].trim();
          if (!mu.props[propName]) mu.props[propName] = {};
          mu.props[propName].dec = propOdds;
        }
      }
      continue;
    }

    // Check for matchup ID in the row
    const muId = row.id?.match(/mu-(\d+)/)?.[1];
    if (muId) {
      currentMatchupId = muId;
      currentFighterIdx = 0;
      if (!matchups[muId]) matchups[muId] = { fighterA: null, fighterB: null, props: {} };
    }

    // Extract fighter name
    const nameSpan = row.querySelector('span.t-b-fcc');
    if (!nameSpan || !currentMatchupId) continue;

    const name = nameSpan.textContent.trim();
    const odds = getConsensusOdds(oddsRow, validColumns);
    const winPct = odds !== null ? oddsToProb(odds) : null;

    if (currentFighterIdx === 0) {
      matchups[currentMatchupId].fighterA = { name, odds, winPct };
    } else {
      matchups[currentMatchupId].fighterB = { name, odds, winPct };
    }
    currentFighterIdx++;
  }

  // Build fighter array from matchups
  for (const muId of Object.keys(matchups)) {
    const mu = matchups[muId];
    if (!mu.fighterA || !mu.fighterB) continue;

    for (const fighter of [mu.fighterA, mu.fighterB]) {
      const entry = { name: fighter.name };

      // Moneyline win probability (replaces DRatings)
      if (fighter.winPct !== null) {
        entry.bfoWinPct = Math.round(fighter.winPct * 100) / 100;
      }

      // Method props: check if this fighter has KO/SUB/DEC odds
      // BFO prop labels use short names ("Ulberg") not full names ("Carlos Ulberg"),
      // so match by checking if any prop key is contained in the fighter name or vice versa
      let propData = mu.props[fighter.name];
      if (!propData) {
        // Fuzzy match: find prop key that matches the fighter's last name or partial name
        const nameLower = fighter.name.toLowerCase();
        for (const [propName, data] of Object.entries(mu.props)) {
          const propLower = propName.toLowerCase();
          if (nameLower.includes(propLower) || propLower.includes(nameLower)) {
            propData = data;
            break;
          }
        }
      }
      if (propData && (propData.ko || propData.sub || propData.dec)) {
        // Convert each method odds to implied probability
        const rawKO = propData.ko ? oddsToProb(propData.ko) : 0;
        const rawSub = propData.sub ? oddsToProb(propData.sub) : 0;
        const rawDec = propData.dec ? oddsToProb(propData.dec) : 0;
        const rawTotal = rawKO + rawSub + rawDec;

        // Normalize to remove vig (sum to 100)
        if (rawTotal > 0) {
          entry.bfoMethodKO = Math.round(rawKO / rawTotal * 10000) / 100;
          entry.bfoMethodSub = Math.round(rawSub / rawTotal * 10000) / 100;
          entry.bfoMethodDec = Math.round(rawDec / rawTotal * 10000) / 100;
          console.log(`[UFC Scraper] ${fighter.name} method props: KO ${entry.bfoMethodKO}%, SUB ${entry.bfoMethodSub}%, DEC ${entry.bfoMethodDec}%`);
        }
      }

      fighters.push(entry);
      console.log(`[UFC Scraper] Found: ${fighter.name} winPct=${entry.bfoWinPct || 'N/A'}%`);
    }
  }

  console.log('[UFC Scraper] Total fighters found:', fighters.length);
  return { source: 'BestFightOdds', fighters };
}

function scrapeFightMatrix() {
  console.log('[UFC Scraper] Running FightMatrix scraper...');
  const fighters = [];
  const seenNames = new Set();

  const fullText = document.body.innerText;
  console.log('[UFC Scraper] Page text sample:', fullText.substring(0, 1500));

  // Helper to match name loosely
  const nameMatches = (fullName, partialName) => {
    const full = fullName.toLowerCase();
    const partial = partialName.toLowerCase().trim();
    const lastName = partial.split(' ').pop();
    return full.includes(lastName) || partial.includes(full.split(' ').pop());
  };

  // Find all matchup lines first using global regex
  // Patterns to handle:
  // [#5][#5UFC] Justin Gaethje (26-5-0, +191) vs. [#8][#7UFC] Paddy Pimblett (23-3-0, -235)
  // [#8P4P][#4][#4UFC] Natalia Silva (19-5-1, -423) vs. [#10P4P][#5][#5UFC] Rose Namajunas (14-7-0, +315)
  // [*] Arnold Allen (20-3-0, +225) vs. [#18][#13UFC] Jean Silva (16-3-0, -279)
  // [#7][#6UFC] Benoit St. Denis (16-3-0, -324) vs. [#8][#7UFC] Dan Hooker (24-13-0, +252)
  // Updated pattern: rankings can have P4P suffix, multiple brackets, or [*] for unranked
  // Name pattern includes period for abbreviations like "St." or "Jr."
  const matchupPattern = /(?:\[#(\d+)\w*\]|\[\*\])(?:\[#\d+\w*\])*\s*([A-Za-zÀ-ÿ\-'.]+(?:\s+[A-Za-zÀ-ÿ\-'.]+)+)\s*\((\d+-\d+-\d+),?\s*([+\-]?\d+)?\)\s*vs\.?\s*(?:\[#(\d+)\w*\]|\[\*\])(?:\[#\d+\w*\])*\s*([A-Za-zÀ-ÿ\-'.]+(?:\s+[A-Za-zÀ-ÿ\-'.]+)+)\s*\((\d+-\d+-\d+),?\s*([+\-]?\d+)?\)/gi;

  let matchupMatch;
  const matchups = [];

  while ((matchupMatch = matchupPattern.exec(fullText)) !== null) {
    matchups.push({
      fullMatch: matchupMatch[0],
      index: matchupMatch.index,
      fighter1: {
        name: matchupMatch[2].trim(),
        ranking: matchupMatch[1] ? parseInt(matchupMatch[1]) : null, // null for [*] unranked
        record: matchupMatch[3],
        bettingOdds: matchupMatch[4] ? parseInt(matchupMatch[4]) : null
      },
      fighter2: {
        name: matchupMatch[6].trim(),
        ranking: matchupMatch[5] ? parseInt(matchupMatch[5]) : null, // null for [*] unranked
        record: matchupMatch[7],
        bettingOdds: matchupMatch[8] ? parseInt(matchupMatch[8]) : null
      }
    });
    console.log('[UFC Scraper] Found matchup:', matchupMatch[2].trim(), 'vs', matchupMatch[6].trim());
  }

  console.log('[UFC Scraper] Total matchups found:', matchups.length);

  // Process each matchup and find associated data in the text that follows
  for (let i = 0; i < matchups.length; i++) {
    const matchup = matchups[i];
    const startIdx = matchup.index;
    const endIdx = (i + 1 < matchups.length) ? matchups[i + 1].index : fullText.length;
    const block = fullText.substring(startIdx, endIdx);

    console.log('[UFC Scraper] Processing block for:', matchup.fighter1.name, 'vs', matchup.fighter2.name);

    const fighter1 = matchup.fighter1;
    const fighter2 = matchup.fighter2;

    // Extract ages: Fighter Ages on Fight Day: Benoit St. Denis 30.1, Dan Hooker 35.9
    // Name pattern includes period for abbreviations like "St." or "Jr."
    const agePattern = /Fighter Ages[^:]*:\s*([A-Za-zÀ-ÿ\-'.\s]+?)\s+([\d.]+),?\s*([A-Za-zÀ-ÿ\-'.\s]+?)\s+([\d.]+)/i;
    const ageMatch = block.match(agePattern);
    if (ageMatch) {
      if (nameMatches(fighter1.name, ageMatch[1])) {
        fighter1.age = parseFloat(ageMatch[2]);
        fighter2.age = parseFloat(ageMatch[4]);
      } else {
        fighter1.age = parseFloat(ageMatch[4]);
        fighter2.age = parseFloat(ageMatch[2]);
      }
    }

    // Extract days since last fight
    const daysPattern = /Days Since Last[^:]*:\s*([A-Za-zÀ-ÿ\-'.\s]+?)\s+(\d+),?\s*([A-Za-zÀ-ÿ\-'.\s]+?)\s+(\d+)/i;
    const daysMatch = block.match(daysPattern);
    if (daysMatch) {
      if (nameMatches(fighter1.name, daysMatch[1])) {
        fighter1.daysSinceLastFight = parseInt(daysMatch[2]);
        fighter2.daysSinceLastFight = parseInt(daysMatch[4]);
      } else {
        fighter1.daysSinceLastFight = parseInt(daysMatch[4]);
        fighter2.daysSinceLastFight = parseInt(daysMatch[2]);
      }
    }

    // Extract rating systems from table
    // Extract rating systems from table - UPDATED to handle tabs/spaces better
    const ratingSystems = {};

    // Elo K170
    const eloK170Pattern = /Elo K170[\t\s]+([A-Za-zÀ-ÿ\-'.\s]+?)[\t\s]+([\d.]+)[\t\s]+([+\-]?[\d.]+)[\t\s]+([\d.]+)%/i;
    const eloK170Match = block.match(eloK170Pattern);
    if (eloK170Match) {
      ratingSystems.eloK170 = {
        favorite: eloK170Match[1].trim(),
        rating: parseFloat(eloK170Match[2]),
        diff: parseFloat(eloK170Match[3]),
        winPct: parseFloat(eloK170Match[4])
      };
    }

    // Elo Modified
    const eloModPattern = /Elo Modified[\t\s]+([A-Za-zÀ-ÿ\-'.\s]+?)[\t\s]+([\d.]+)[\t\s]+([+\-]?[\d.]+)[\t\s]+([\d.]+)%/i;
    const eloModMatch = block.match(eloModPattern);
    if (eloModMatch) {
      ratingSystems.eloMod = {
        favorite: eloModMatch[1].trim(),
        rating: parseFloat(eloModMatch[2]),
        diff: parseFloat(eloModMatch[3]),
        winPct: parseFloat(eloModMatch[4])
      };
    }

    // Glicko-1
    const glickoPattern = /Glicko-1[\t\s]+([A-Za-zÀ-ÿ\-'.\s]+?)[\t\s]+([\d.]+)[\t\s]+([+\-]?[\d.]+)[\t\s]+([\d.]+)%/i;
    const glickoMatch = block.match(glickoPattern);
    if (glickoMatch) {
      ratingSystems.glicko = {
        favorite: glickoMatch[1].trim(),
        rating: parseFloat(glickoMatch[2]),
        diff: parseFloat(glickoMatch[3]),
        winPct: parseFloat(glickoMatch[4])
      };
    }

    // WHR (Whole-History Rating)
    const whrPattern = /WHR[\t\s]+([A-Za-zÀ-ÿ\-'.\s]+?)[\t\s]+([\d.]+)[\t\s]+([+\-]?[\d.]+)[\t\s]+([\d.]+)%/i;
    const whrMatch = block.match(whrPattern);
    if (whrMatch) {
      ratingSystems.whr = {
        favorite: whrMatch[1].trim(),
        rating: parseFloat(whrMatch[2]),
        diff: parseFloat(whrMatch[3]),
        winPct: parseFloat(whrMatch[4])
      };
    }

    // Betting Odds from table
    const oddsPattern = /Betting Odds[\t\s]+([A-Za-zÀ-ÿ\-'.\s]+?)[\t\s]+([+\-]?\d+)[\t\s]+([\d.]+)%/i;
    const oddsMatch = block.match(oddsPattern);
    if (oddsMatch) {
      ratingSystems.bettingOdds = {
        favorite: oddsMatch[1].trim(),
        odds: parseInt(oddsMatch[2]),
        winPct: parseFloat(oddsMatch[3])
      };
    }

    // Extract last 3 fight records
    const last3Pattern = /Last 3 Fights:\s*([A-Za-zÀ-ÿ\-'.\s]+?)\s*\((\d+-\d+-\d+)\)/gi;
    let last3Match;
    while ((last3Match = last3Pattern.exec(block)) !== null) {
      if (nameMatches(fighter1.name, last3Match[1])) {
        fighter1.last3Record = last3Match[2];
      } else if (nameMatches(fighter2.name, last3Match[1])) {
        fighter2.last3Record = last3Match[2];
      }
    }

    // Assign rating data to fighters
    const assignRatingToFighters = (system, data) => {
      if (!data) return;
      const isFighter1Fav = nameMatches(fighter1.name, data.favorite);

      if (isFighter1Fav) {
        fighter1[system] = { rating: data.rating, diff: data.diff, winPct: data.winPct };
        fighter2[system] = { rating: data.rating - data.diff, diff: -data.diff, winPct: 100 - data.winPct };
      } else {
        fighter2[system] = { rating: data.rating, diff: data.diff, winPct: data.winPct };
        fighter1[system] = { rating: data.rating - data.diff, diff: -data.diff, winPct: 100 - data.winPct };
      }
    };

    assignRatingToFighters('eloK170', ratingSystems.eloK170);
    assignRatingToFighters('eloMod', ratingSystems.eloMod);
    assignRatingToFighters('glicko', ratingSystems.glicko);
    assignRatingToFighters('whr', ratingSystems.whr);

    // Handle betting odds
    if (ratingSystems.bettingOdds) {
      const isFighter1Fav = nameMatches(fighter1.name, ratingSystems.bettingOdds.favorite);
      if (isFighter1Fav) {
        fighter1.bettingWinPct = ratingSystems.bettingOdds.winPct;
        fighter2.bettingWinPct = 100 - ratingSystems.bettingOdds.winPct;
      } else {
        fighter2.bettingWinPct = ratingSystems.bettingOdds.winPct;
        fighter1.bettingWinPct = 100 - ratingSystems.bettingOdds.winPct;
      }
    }

    // Use Elo K170 rating as the primary CIRRS value for backwards compatibility
    if (fighter1.eloK170) fighter1.cirrs = Math.round(fighter1.eloK170.rating);
    if (fighter2.eloK170) fighter2.cirrs = Math.round(fighter2.eloK170.rating);

    // Add fighters to list
    const key1 = fighter1.name.toLowerCase();
    const key2 = fighter2.name.toLowerCase();

    if (!seenNames.has(key1)) {
      seenNames.add(key1);
      fighters.push(fighter1);
      console.log('[UFC Scraper] Added fighter:', fighter1.name, fighter1);
    }

    if (!seenNames.has(key2)) {
      seenNames.add(key2);
      fighters.push(fighter2);
      console.log('[UFC Scraper] Added fighter:', fighter2.name, fighter2);
    }
  }

  // Fallback: Simple pattern for basic CIRRS extraction if detailed parsing fails
  if (fighters.length === 0) {
    console.log('[UFC Scraper] Falling back to simple pattern matching...');
    const simplePattern = /([A-Z][a-z]+\s+[A-Z][a-zA-Z\-']+)\s+(\d{4})/g;
    let match;
    while ((match = simplePattern.exec(fullText)) !== null) {
      const name = match[1].trim();
      const cirrs = parseInt(match[2]);
      const key = name.toLowerCase();

      if (!seenNames.has(key) && cirrs >= 1000 && cirrs <= 2500) {
        seenNames.add(key);
        fighters.push({ name, cirrs });
        console.log('[UFC Scraper] Found via simple pattern:', name, 'CIRRS:', cirrs);
      }
    }
  }

  console.log('[UFC Scraper] Total found:', fighters.length);
  return { source: 'FightMatrix', fighters };
}

// ==================== RESULTS SCRAPERS ====================

function scrapeTapologyResults() {
  console.log('[UFC Scraper] Running Tapology results scraper...');
  const results = [];
  const seenFights = new Set();

  const fullText = document.body.innerText;
  const lines = fullText.split('\n').map(l => l.trim()).filter(l => l.length > 0);

  // Debug: log single-char lines and sample around "def" to understand page format
  const singleCharLines = lines.map((l, i) => [l, i]).filter(([l]) => l.length === 1 && /[WLDR]/.test(l));
  console.log('[UFC Scraper] Single-char W/L/D lines:', singleCharLines.slice(0, 30).map(([l, i]) => `${i}:"${l}" before:"${lines[i-1]}" after:"${lines[i+1]}"`));

  const defPositions = [];
  let searchPos = 0;
  while ((searchPos = fullText.indexOf(' def ', searchPos)) >= 0) {
    defPositions.push(searchPos);
    const ctx = fullText.substring(Math.max(0, searchPos - 40), searchPos + 40).replace(/\n/g, '|');
    console.log('[UFC Scraper] "def" found at', searchPos, ':', ctx);
    searchPos += 5;
  }

  // ===== STRATEGY 1: "def" pattern (primary - works on real Tapology pages) =====
  const namePattern = '(?:[A-Z]\\.?\\s*[A-Z][a-zA-Zà-ÿ\\-\']+(?:\\s+[A-Z][a-zA-Zà-ÿ\\-\']+)*|[A-Z][a-zA-Zà-ÿ\\-\']+(?:\\s+[A-Z][a-zA-Zà-ÿ\\-\']+)+|[A-Z][a-zA-Zà-ÿ]{4,})';
  // Allow optional period/punctuation after name (handles "Jr.", "III.", etc.)
  const defPattern = new RegExp('(' + namePattern + ')\\.?\\s+def\\.?\\s+(' + namePattern + ')', 'g');

  let match;
  while ((match = defPattern.exec(fullText)) !== null) {
    // Clean names: strip W/L/D markers that get captured due to \s matching \n
    let winner = match[1].trim().replace(/^[WLD]\n\s*/g, '').replace(/^[WLD]\s(?=[A-Z])/, '').trim();
    let loser = match[2].trim().replace(/^[WLD]\n\s*/g, '').replace(/^[WLD]\s(?=[A-Z])/, '').trim();

    if (winner.length < 3 || loser.length < 3) continue;

    const key = winner.toLowerCase() + '-' + loser.toLowerCase();
    if (seenFights.has(key)) continue;
    seenFights.add(key);

    // Method detection: look in text BEFORE the def match (method appears above fighter names)
    let beforeText = fullText.substring(Math.max(0, match.index - 400), match.index);
    // Truncate at previous "def" to avoid crossing into previous fight
    const prevDef = beforeText.lastIndexOf(' def ');
    if (prevDef >= 0) {
      beforeText = beforeText.substring(prevDef + 5);
    }

    let method = 'DEC';
    let round = 'DEC';

    // Scan backward through lines before the match for method indicators
    const beforeLines = beforeText.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    for (let k = beforeLines.length - 1; k >= 0; k--) {
      const line = beforeLines[k];
      // Skip prediction header rows
      if (/KO\/TKO\s+Submission\s+Decision/i.test(line)) continue;

      if (/KO\/TKO/i.test(line) || (/\bTKO\b/i.test(line) && !/Submission/i.test(line))) {
        method = 'KO';
        break;
      }
      if (/Submission|Technical Sub/i.test(line)) {
        method = 'SUB';
        break;
      }
      if (/Decision|Unanimous|Majority|Split/i.test(line)) {
        method = 'DEC';
        break;
      }
      if (/Ends in a Draw/i.test(line)) {
        method = 'DRAW';
        break;
      }
    }

    // Also check text AFTER def as fallback for method (some page layouts)
    if (method === 'DEC') {
      let afterText = fullText.substring(match.index + match[0].length, match.index + match[0].length + 200);
      const nextDef = afterText.indexOf(' def ');
      if (nextDef > 0) afterText = afterText.substring(0, nextDef);
      // Only use after-text if it has a clear non-DEC indicator
      if (/KO\/TKO/i.test(afterText) && !/Decision|Unanimous/i.test(afterText.substring(0, afterText.search(/KO\/TKO/i)))) {
        method = 'KO';
      } else if (/Submission/i.test(afterText) && !/Decision|Unanimous/i.test(afterText.substring(0, afterText.search(/Submission/i)))) {
        method = 'SUB';
      }
    }

    // Extract round from beforeText (find last "Round X of Y" closest to this fight)
    if (method !== 'DEC') {
      const roundMatches = [...beforeText.matchAll(/Round\s*(\d+)\s*of\s*\d+/gi)];
      if (roundMatches.length > 0) {
        round = 'R' + roundMatches[roundMatches.length - 1][1];
      }
    }

    results.push({ winner, loser, method, round });
    console.log('[UFC Scraper] [def] Result:', winner, 'def', loser, '-', method, round);
  }

  // ===== STRATEGY 2: W/L markers (supplement - catches fights missed by def) =====
  for (let i = 1; i < lines.length - 1; i++) {
    if (lines[i] !== 'W') continue;

    const nameBefore = lines[i - 1];
    const nameAfter = lines[i + 1];
    if (!nameBefore || !nameAfter) continue;
    if (nameBefore.length < 3 || nameAfter.length < 3) continue;
    if (/^\d/.test(nameBefore)) continue;
    if (nameBefore.toLowerCase() !== nameAfter.toLowerCase()) continue;

    const winnerName = nameBefore;

    // Find corresponding "L" marker
    let loserName = null;
    for (let j = i + 2; j < Math.min(i + 25, lines.length - 1); j++) {
      if (lines[j] === 'L') {
        loserName = lines[j + 1];
        break;
      }
      if (lines[j] === 'W' && j > 0 && j + 1 < lines.length &&
          lines[j - 1].length >= 3 && lines[j + 1].length >= 3 &&
          lines[j - 1].toLowerCase() === lines[j + 1].toLowerCase()) {
        break;
      }
    }

    if (!loserName || loserName.length < 3 || /^\d/.test(loserName)) continue;

    // Skip if already found by def pattern
    const key = winnerName.toLowerCase() + '-' + loserName.toLowerCase();
    if (seenFights.has(key)) continue;
    seenFights.add(key);

    // Method detection: look before winner in lines
    let method = 'DEC';
    let round = 'DEC';
    for (let k = i - 2; k >= Math.max(0, i - 8); k--) {
      const line = lines[k];
      if (/KO\/TKO\s+Submission\s+Decision/i.test(line)) continue;
      if (/KO\/TKO/i.test(line)) { method = 'KO'; break; }
      if (/Submission|Technical Sub/i.test(line)) { method = 'SUB'; break; }
      if (/Decision|Unanimous|Majority|Split/i.test(line)) { method = 'DEC'; break; }
    }
    if (method !== 'DEC') {
      for (let k = i - 2; k >= Math.max(0, i - 8); k--) {
        const rm = lines[k].match(/Round\s*(\d+)\s*of\s*\d+/i);
        if (rm) { round = 'R' + rm[1]; break; }
      }
    }

    results.push({ winner: winnerName, loser: loserName, method, round });
    console.log('[UFC Scraper] [W/L] Result:', winnerName, 'def', loserName, '-', method, round);
  }

  // ===== DRAWS =====
  // Strategy A: D markers (standalone D with matching names)
  const processedDrawIndices = new Set();
  for (let i = 1; i < lines.length - 1; i++) {
    if (lines[i] !== 'D' || processedDrawIndices.has(i)) continue;

    const nameBefore = lines[i - 1];
    const nameAfter = lines[i + 1];
    if (!nameBefore || !nameAfter) continue;
    if (nameBefore.length < 3 || nameAfter.length < 3) continue;
    if (/^\d/.test(nameBefore)) continue;
    if (nameBefore.toLowerCase() !== nameAfter.toLowerCase()) continue;

    const fighter1 = nameBefore;
    processedDrawIndices.add(i);

    for (let j = i + 2; j < Math.min(i + 25, lines.length - 1); j++) {
      if (lines[j] === 'D') {
        const fighter2 = lines[j + 1];
        if (fighter2 && fighter2.length >= 3 && !/^\d/.test(fighter2)) {
          const key = fighter1.toLowerCase() + '-' + fighter2.toLowerCase();
          if (!seenFights.has(key)) {
            seenFights.add(key);
            processedDrawIndices.add(j);
            results.push({
              winner: 'DRAW',
              loser: fighter1 + ' vs ' + fighter2,
              method: 'DRAW',
              round: 'DEC'
            });
            console.log('[UFC Scraper] [D] Draw:', fighter1, 'vs', fighter2);
          }
        }
        break;
      }
      if (lines[j] === 'W' && j > 0 && j + 1 < lines.length &&
          lines[j - 1].toLowerCase() === lines[j + 1]?.toLowerCase()) {
        break;
      }
    }
  }

  // Strategy B: "Draw" text near "vs" pattern (fallback for draws)
  const drawPattern = /([A-Z][a-zA-Zà-ÿ\-']+(?:\s+[A-Z][a-zA-Zà-ÿ\-']+)+)\s+vs\.?\s+([A-Z][a-zA-Zà-ÿ\-']+(?:\s+[A-Z][a-zA-Zà-ÿ\-']+)+)/g;
  while ((match = drawPattern.exec(fullText)) !== null) {
    const beforeCtx = fullText.substring(Math.max(0, match.index - 100), match.index);
    const afterCtx = fullText.substring(match.index, match.index + match[0].length + 100);
    if (/FIGHT OF THE NIGHT|PERFORMANCE OF THE NIGHT|BONUS|Bonuses|Awards/i.test(beforeCtx + afterCtx)) continue;
    if (/Cancelled|Fizzled/i.test(beforeCtx + afterCtx)) continue;

    if (/Draw|NC|No Contest/i.test(beforeCtx + afterCtx) && !/def\s/i.test(afterCtx)) {
      const fighter1 = match[1].trim();
      const fighter2 = match[2].trim();
      const key = fighter1.toLowerCase() + '-' + fighter2.toLowerCase();
      const keyRev = fighter2.toLowerCase() + '-' + fighter1.toLowerCase();
      if (!seenFights.has(key) && !seenFights.has(keyRev)) {
        seenFights.add(key);
        results.push({
          winner: 'DRAW',
          loser: fighter1 + ' vs ' + fighter2,
          method: 'DRAW',
          round: 'DEC'
        });
        console.log('[UFC Scraper] [vs] Draw:', fighter1, 'vs', fighter2);
      }
    }
  }

  // ===== CANCELLED FIGHTS =====
  const cancelledSection = fullText.match(/Cancelled\s*(?:&|and)?\s*Fizzled\s*Bouts[\s\S]*?(?=(?:Additional|Weigh|Awards|$))/i);
  if (cancelledSection) {
    console.log('[UFC Scraper] Found cancelled section');
    const cancelledText = cancelledSection[0];

    const cancelledFilterWords = ['fizzled', 'bouts', 'cancelled', 'withdrew', 'medical', 'injury', 'visa', 'weight', 'rescheduled', 'ruptured', 'acl', 'missed', 'broken', 'healed', 'issues'];

    const cancelledLines = cancelledText.split('\n');
    const vsPattern = /([A-Z][a-zA-Zà-ÿ\-']+(?:\s+[A-Z][a-zA-Zà-ÿ\-']+)+)\s+vs\.?\s+([A-Z][a-zA-Zà-ÿ\-']+(?:\s+[A-Z][a-zA-Zà-ÿ\-']+)+)/i;

    for (const line of cancelledLines) {
      const cancelMatch = vsPattern.exec(line);
      if (cancelMatch) {
        const fighter1 = cancelMatch[1].trim();
        const fighter2 = cancelMatch[2].trim();

        if (fighter1.length < 5 || fighter2.length < 5) continue;
        const f1Lower = fighter1.toLowerCase();
        const f2Lower = fighter2.toLowerCase();
        if (cancelledFilterWords.some(w => f1Lower.includes(w) || f2Lower.includes(w))) continue;

        const key = f1Lower + '-' + f2Lower;
        if (!seenFights.has(key)) {
          seenFights.add(key);
          results.push({
            fighterA: fighter1,
            fighterB: fighter2,
            cancelled: true,
            method: 'CANCELLED'
          });
          console.log('[UFC Scraper] Cancelled fight:', fighter1, 'vs', fighter2);
        }
      }
    }
  }

  console.log('[UFC Scraper] Total results found:', results.length);
  return { source: 'Tapology', results };
}

function scrapeUFCResults() {
  console.log('[UFC Scraper] Running UFC.com results scraper...');
  const results = [];

  // UFC.com has structured fight result cards
  const fightCards = document.querySelectorAll('.c-listing-fight, .l-listing__item');

  fightCards.forEach(card => {
    // Find winner (usually has a "Winner" badge or different styling)
    const corners = card.querySelectorAll('.c-listing-fight__corner');
    let winner = null;
    let loser = null;

    corners.forEach(corner => {
      const nameEl = corner.querySelector('.c-listing-fight__corner-name a, .c-listing-fight__corner-name');
      const name = nameEl?.textContent?.trim();

      // Check if this corner won
      const isWinner = corner.classList.contains('winner') ||
        corner.querySelector('.c-listing-fight__outcome--Winner') ||
        corner.querySelector('[class*="winner"]');

      if (isWinner) {
        winner = name;
      } else if (name) {
        loser = name;
      }
    });

    // Get method and round
    const detailsEl = card.querySelector('.c-listing-fight__result-text, .c-listing-fight__details');
    const detailsText = detailsEl?.textContent?.trim() || '';

    let method = 'DEC';
    if (detailsText.toLowerCase().includes('ko') || detailsText.toLowerCase().includes('tko')) {
      method = 'KO';
    } else if (detailsText.toLowerCase().includes('sub')) {
      method = 'SUB';
    }

    let round = 'DEC';
    const roundMatch = detailsText.match(/R(\d)|Round\s*(\d)/i);
    if (roundMatch && method !== 'DEC') {
      round = 'R' + (roundMatch[1] || roundMatch[2]);
    }

    if (winner && loser) {
      results.push({ winner, loser, method, round });
      console.log('[UFC Scraper] Result:', winner, 'def.', loser, 'via', method, round);
    }
  });

  // Fallback: parse text for "def." patterns
  if (results.length === 0) {
    const fullText = document.body.innerText;
    const defPattern = /([A-Z][a-zA-Z\-'\s]+)\s+def\.\s+([A-Z][a-zA-Z\-'\s]+)/gi;
    let match;

    while ((match = defPattern.exec(fullText)) !== null) {
      const winner = match[1].trim();
      const loser = match[2].trim();

      if (winner.length > 3 && loser.length > 3) {
        results.push({ winner, loser, method: 'DEC', round: 'DEC' });
        console.log('[UFC Scraper] Result from text:', winner, 'def.', loser);
      }
    }
  }

  console.log('[UFC Scraper] Total results found:', results.length);
  return { source: 'UFC', results };
}
