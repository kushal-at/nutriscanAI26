/**
 * NutriScanAI — Production Frontend Application Controller
 * Dual-View Food Diary, Live Camera Capture, LocalStorage & Machine Learning Integration
 */

(function () {
    'use strict';

    // -------------------------------------------------------------
    // CONSTANTS & STORAGE KEYS (With Backward Compatibility)
    // -------------------------------------------------------------
    const STORAGE_KEY_MEALS = 'nutriscan_meals';
    const STORAGE_KEY_GOALS = 'nutriscan_goals';
    const STORAGE_KEY_PROFILE = 'nutriscan_profile';
    const STORAGE_KEY_WATER = 'nutriscan_water';
    const STORAGE_KEY_THEME = 'nutriscan_theme';

    // Dynamic Origin Resolution: Binds cleanly to current window host (127.0.0.1, localhost, or LAN)
    function resolveServiceUrl(defaultPort) {
        if (typeof window !== 'undefined' && window.location) {
            const proto = (window.location.protocol && window.location.protocol.startsWith('http')) 
                ? window.location.protocol 
                : 'http:';
            const host = window.location.hostname || '127.0.0.1';
            return `${proto}//${host}:${defaultPort}`;
        }
        return `http://127.0.0.1:${defaultPort}`;
    }

    const BACKEND_BASE_URL = window.BACKEND_API_URL || resolveServiceUrl(8000);
    const VISION_BASE_URL = window.VISION_API_URL || resolveServiceUrl(8001);

    // -------------------------------------------------------------
    // DEFAULT SEED DATA
    // -------------------------------------------------------------
    function getTodayDateString(offsetDays = 0) {
        const d = new Date();
        d.setDate(d.getDate() - offsetDays);
        return d.toISOString().split('T')[0];
    }

    const DEFAULT_BASELINE_GOALS = {
        calories: 2000,
        protein: 100,
        carbs: 220,
        fat: 65,
        water: 2.5,
        streak: 1,
        dietaryPatterns: [],
        allergens: "None specified",
        schedule: "Breakfast: 8:30 AM · Lunch: 1:00 PM · Dinner: 7:30 PM"
    };

    const DEFAULT_MEALS = [];

    // -------------------------------------------------------------
    // DATA REPOSITORY (LOCAL STORAGE WITH MIGRATION)
    // -------------------------------------------------------------
    const Storage = {
        getMeals() {
            try {
                let raw = localStorage.getItem(STORAGE_KEY_MEALS);
                if (!raw) {
                    raw = localStorage.getItem('nutri_meals');
                }
                if (!raw) {
                    return [];
                }
                const parsed = JSON.parse(raw);
                // If legacy dummy seed meals exist without an onboarded profile, clear them
                if (!localStorage.getItem('nutriscan_user_mode') && Array.isArray(parsed) && parsed.some(m => m.id && m.id.startsWith('seed-'))) {
                    return [];
                }
                return Array.isArray(parsed) ? parsed : [];
            } catch (e) {
                console.error("Error reading meals from localStorage", e);
                return [];
            }
        },

        saveMeals(meals) {
            try {
                localStorage.setItem(STORAGE_KEY_MEALS, JSON.stringify(meals));
            } catch (e) {
                console.error("Error saving meals to localStorage", e);
                if (e.name === 'QuotaExceededError' || e.code === 22) {
                    try {
                        const trimmed = meals.map((m, idx) => {
                            if (idx > 5 && m.image && m.image.startsWith('data:')) {
                                return { ...m, image: "assets/nourish-bowl.png" };
                            }
                            return m;
                        });
                        localStorage.setItem(STORAGE_KEY_MEALS, JSON.stringify(trimmed));
                    } catch (e2) {
                        showToast("Local storage full. Older meal image caches were cleared.", true);
                    }
                }
            }
        },

        addMeal(meal) {
            const meals = this.getMeals();
            meals.unshift(meal);
            this.saveMeals(meals);
            if (window.NutriScanSync && typeof window.NutriScanSync.sendMealToCloud === 'function') {
                window.NutriScanSync.sendMealToCloud(meal);
            }
            return meals;
        },

        deleteMeal(id) {
            const meals = this.getMeals().filter(m => m.id !== id);
            this.saveMeals(meals);
            if (window.NutriScanSync && typeof window.NutriScanSync.deleteMealFromCloud === 'function') {
                window.NutriScanSync.deleteMealFromCloud(id);
            }
            return meals;
        },

        getGoals() {
            try {
                let raw = localStorage.getItem(STORAGE_KEY_GOALS) || localStorage.getItem('nutri_goals');
                if (!raw) {
                    return { ...DEFAULT_BASELINE_GOALS };
                }
                return { ...DEFAULT_BASELINE_GOALS, ...JSON.parse(raw) };
            } catch (e) {
                return { ...DEFAULT_BASELINE_GOALS };
            }
        },

        saveGoals(goals) {
            try {
                localStorage.setItem(STORAGE_KEY_GOALS, JSON.stringify(goals));
                if (window.NutriScanSync && typeof window.NutriScanSync.sendGoalsToCloud === 'function') {
                    window.NutriScanSync.sendGoalsToCloud(goals);
                }
            } catch (e) {
                console.error("Error saving goals", e);
            }
        },

        getProfile() {
            try {
                let raw = localStorage.getItem(STORAGE_KEY_PROFILE) || localStorage.getItem('nutri_profile');
                if (!raw) {
                    return null; // No profile when fresh device!
                }
                const parsed = JSON.parse(raw);
                // If legacy dummy profile exists without onboarded flag, ignore it
                if (!localStorage.getItem('nutriscan_user_mode') && parsed && parsed.name === 'Alex Morgan') {
                    return null;
                }
                return parsed;
            } catch (e) {
                return null;
            }
        },

        saveProfile(profile) {
            try {
                localStorage.setItem(STORAGE_KEY_PROFILE, JSON.stringify(profile));
                if (window.NutriScanSync && typeof window.NutriScanSync.sendProfileToCloud === 'function') {
                    window.NutriScanSync.sendProfileToCloud(profile);
                }
            } catch (e) {
                console.error("Error saving profile", e);
            }
        },

        getWater() {
            try {
                const todayStr = getTodayDateString(0);
                const raw = localStorage.getItem(STORAGE_KEY_WATER);
                if (!raw) {
                    return { date: todayStr, ml: 0 };
                }
                const parsed = JSON.parse(raw);
                if (parsed.date !== todayStr) {
                    const fresh = { date: todayStr, ml: 0 };
                    localStorage.setItem(STORAGE_KEY_WATER, JSON.stringify(fresh));
                    return fresh;
                }
                return parsed;
            } catch (e) {
                return { date: getTodayDateString(0), ml: 0 };
            }
        },

        saveWater(ml) {
            try {
                const data = { date: getTodayDateString(0), ml: Math.max(0, ml) };
                localStorage.setItem(STORAGE_KEY_WATER, JSON.stringify(data));
                if (window.NutriScanSync && typeof window.NutriScanSync.sendWaterToCloud === 'function') {
                    window.NutriScanSync.sendWaterToCloud(data.date, data.ml);
                }
            } catch (e) {
                console.error("Error saving water", e);
            }
        },

        resetAll() {
            localStorage.removeItem(STORAGE_KEY_MEALS);
            localStorage.removeItem(STORAGE_KEY_GOALS);
            localStorage.removeItem(STORAGE_KEY_PROFILE);
            localStorage.removeItem(STORAGE_KEY_WATER);
            localStorage.removeItem('nutriscan_user_mode');
            localStorage.removeItem('nutriscan_local_onboarded');
            localStorage.removeItem('nutri_meals');
            localStorage.removeItem('nutri_profile');
            localStorage.removeItem('nutri_goals');
            if (window.checkLoginState) window.checkLoginState();
        }
    };

    window.NutriScanStorage = Storage;

    // -------------------------------------------------------------
    // SCIENTIFIC BMR / TDEE CALCULATOR (Mifflin-St Jeor Formula)
    // -------------------------------------------------------------
    function calculateTDEE(weightKg, heightCm, ageYears, gender, activityLevel, goal) {
        // Basal Metabolic Rate
        let bmr = (10 * weightKg) + (6.25 * heightCm) - (5 * ageYears);
        if (gender === 'Male') {
            bmr += 5;
        } else {
            bmr -= 161;
        }

        // Activity Multipliers
        const activityMap = {
            sedentary: 1.2,
            light: 1.375,
            moderate: 1.55,
            very: 1.725,
            extra: 1.9
        };
        const multiplier = activityMap[activityLevel] || 1.375;
        let tdee = bmr * multiplier;

        // Goal Adjustment
        if (goal.includes('fat loss') || goal.includes('loss')) {
            tdee -= 400; // Safe moderate deficit
        } else if (goal.includes('gain') || goal.includes('Muscle')) {
            tdee += 350; // Lean caloric surplus
        }

        const calories = Math.max(1200, Math.round(tdee / 50) * 50);

        // Optimal Macro Distribution
        // Protein: 1.8g per kg bodyweight
        const protein = Math.min(220, Math.max(50, Math.round(weightKg * 1.8)));
        // Fat: 0.9g per kg bodyweight
        const fat = Math.min(120, Math.max(35, Math.round(weightKg * 0.9)));
        // Remaining calories from Carbohydrates (4 kcal/g)
        const carbKcal = Math.max(300, calories - (protein * 4 + fat * 9));
        const carbs = Math.max(60, Math.round(carbKcal / 4));

        // Water recommendation (35ml per kg bodyweight)
        const water = Math.max(1.8, Math.min(5.0, Math.round((weightKg * 0.035) * 10) / 10));

        return { calories, protein, carbs, fat, water };
    }

    // -------------------------------------------------------------
    // UI TOAST NOTIFICATION HELPER
    // -------------------------------------------------------------
    function showToast(message, isError = false) {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast ${isError ? 'toast-error' : ''}`;
        toast.innerHTML = `
            <i data-lucide="${isError ? 'alert-triangle' : 'check-circle'}" class="icon-sm"></i>
            <span>${escapeHTML(message)}</span>
        `;
        container.appendChild(toast);
        renderLucide();

        setTimeout(() => {
            toast.style.transition = 'opacity 0.25s, transform 0.25s';
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(10px)';
            setTimeout(() => toast.remove(), 250);
        }, 3200);
    }

    function escapeHTML(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function renderLucide() {
        if (window.lucide && typeof window.lucide.createIcons === 'function') {
            window.lucide.createIcons();
        }
    }

    // -------------------------------------------------------------
    // ACTIVE STATE
    // -------------------------------------------------------------
    let currentView = 'dashboard';
    let uploadedTopFile = null;
    let uploadedSideFile = null;
    let currentScanResult = null;
    let backendOnline = false;

    // Camera state
    let cameraStream = null;
    let cameraFacingMode = 'environment';

    // -------------------------------------------------------------
    // VIEW CONTROLLER
    // -------------------------------------------------------------
    function switchView(viewName) {
        const targetView = document.getElementById(`view-${viewName}`);
        if (!targetView) return;

        currentView = viewName;

        // Hide all views and activate target
        document.querySelectorAll('.view-screen').forEach(el => el.classList.remove('active'));
        targetView.classList.add('active');

        // Update Desktop Sidebar active states
        document.querySelectorAll('.sidebar .navitem').forEach(el => {
            if (el.getAttribute('data-view') === viewName) {
                el.classList.add('active');
            } else {
                el.classList.remove('active');
            }
        });

        // Update Mobile Bottom Nav active states
        document.querySelectorAll('.mobile-bottom-nav .mob-nav-btn, .mobile-bottom-nav .mob-nav-scan').forEach(el => {
            if (el.getAttribute('data-view') === viewName) {
                el.classList.add('active');
            } else {
                el.classList.remove('active');
            }
        });

        // Update Desktop Topbar Breadcrumb
        const crumbEl = document.getElementById('desktop-crumb-page');
        if (crumbEl) {
            const titles = {
                dashboard: 'Dashboard',
                scanner: 'Food Scanner',
                history: 'Meal History',
                reports: 'Diet Reports',
                goals: 'Goals & Targets',
                profile: 'Profile & Settings'
            };
            crumbEl.textContent = titles[viewName] || viewName;
        }

        // Render current view
        if (viewName === 'dashboard') renderDashboard();
        if (viewName === 'scanner') renderScanner();
        if (viewName === 'history') renderHistory();
        if (viewName === 'reports') renderReports();
        if (viewName === 'goals') renderGoals();
        if (viewName === 'profile') renderProfile();

        renderLucide();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // -------------------------------------------------------------
    // THEME ENGINE (LIGHT / DARK)
    // -------------------------------------------------------------
    function initTheme() {
        const savedTheme = localStorage.getItem(STORAGE_KEY_THEME) ||
            (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');

        applyTheme(savedTheme);

        const btnToggle = document.getElementById('btn-theme-toggle');
        if (btnToggle) {
            btnToggle.addEventListener('click', () => {
                const current = document.documentElement.getAttribute('data-theme') || 'light';
                const next = current === 'dark' ? 'light' : 'dark';
                applyTheme(next);
                localStorage.setItem(STORAGE_KEY_THEME, next);
                showToast(`${next === 'dark' ? 'Dark' : 'Light'} theme enabled`);
            });
        }
    }

    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        const icon = document.getElementById('theme-toggle-icon');
        if (icon) {
            icon.setAttribute('data-lucide', theme === 'dark' ? 'sun' : 'moon');
            renderLucide();
        }
    }

    // -------------------------------------------------------------
    // BACKEND SERVICE STATUS CHECKER
    // -------------------------------------------------------------
    async function checkBackendStatus() {
        const pillText = document.getElementById('backend-status-text');
        const pillDot = document.getElementById('backend-status-dot');

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 3500);

            const isPort8000 = (typeof window !== 'undefined' && window.location && window.location.port === "8000");
            const healthUrl = isPort8000 ? "/health" : `${BACKEND_BASE_URL}/health`;

            const resp = await fetch(healthUrl, {
                method: 'GET',
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            backendOnline = resp.ok;
            if (pillText) pillText.textContent = backendOnline ? "AI Online" : "Local Storage";
            if (pillDot) {
                if (backendOnline) pillDot.classList.remove('offline');
                else pillDot.classList.add('offline');
            }
        } catch (e) {
            backendOnline = false;
            if (pillText) pillText.textContent = "Local Storage";
            if (pillDot) pillDot.classList.add('offline');
        }
    }

    // -------------------------------------------------------------
    // SCREEN 1: DASHBOARD RENDERER & WATER TRACKER
    // -------------------------------------------------------------
    function renderDashboard() {
        const meals = Storage.getMeals();
        const goals = Storage.getGoals();
        const profile = Storage.getProfile();
        const todayStr = getTodayDateString(0);

        const todaysMeals = meals.filter(m => m.date === todayStr);

        let consumedKcal = 0;
        let consumedProtein = 0;
        let consumedCarbs = 0;
        let consumedFat = 0;

        todaysMeals.forEach(m => {
            consumedKcal += Number(m.calories) || 0;
            consumedProtein += Number(m.protein) || 0;
            consumedCarbs += Number(m.carbs) || 0;
            consumedFat += Number(m.fat) || 0;
        });

        // Greeting & Streak
        const greetingEl = document.getElementById('dash-greeting');
        if (greetingEl) {
            const firstName = profile?.name ? profile.name.split(' ')[0] : 'there';
            greetingEl.textContent = `Good morning, ${firstName}`;
        }
        const streakEl = document.getElementById('dash-streak-count');
        if (streakEl) streakEl.textContent = `${goals.streak || 5} day streak`;

        // Date string format
        const dateStrEl = document.getElementById('dash-date-str');
        if (dateStrEl) {
            const options = { weekday: 'long', month: 'short', day: 'numeric' };
            dateStrEl.textContent = new Date().toLocaleDateString('en-US', options);
        }

        // Calorie Ring & Values
        const targetKcal = goals.calories || 2000;
        const pct = Math.min(100, Math.round((consumedKcal / targetKcal) * 100));
        const remaining = Math.max(0, targetKcal - consumedKcal);

        const ringEl = document.getElementById('dash-calorie-ring');
        if (ringEl) {
            ringEl.style.setProperty('--ring-pct', `${pct}%`);
        }
        const ringConsumedEl = document.getElementById('dash-ring-consumed');
        if (ringConsumedEl) ringConsumedEl.textContent = consumedKcal.toLocaleString();

        const kcalTextEl = document.getElementById('dash-kcal-text');
        if (kcalTextEl) {
            kcalTextEl.innerHTML = `${consumedKcal.toLocaleString()} <span style="font-size: 13px; font-weight: 600; color: var(--muted);">/ ${targetKcal.toLocaleString()} kcal</span>`;
        }
        const goalPctBadge = document.getElementById('dash-goal-pct-badge');
        if (goalPctBadge) goalPctBadge.textContent = `${pct}% of daily goal`;

        const remEl = document.getElementById('dash-kcal-remaining');
        if (remEl) {
            remEl.textContent = `${remaining.toLocaleString()} kcal remaining today`;
        }

        // Macro Cards
        const pTarget = goals.protein || 100;
        const cTarget = goals.carbs || 220;
        const fTarget = goals.fat || 65;

        const pPct = Math.round((consumedProtein / pTarget) * 100);
        const cPct = Math.round((consumedCarbs / cTarget) * 100);
        const fPct = Math.round((consumedFat / fTarget) * 100);

        document.getElementById('dash-macro-protein').textContent = `${consumedProtein}g`;
        document.getElementById('dash-macro-protein-pct').textContent = `Protein · ${pPct}%`;

        document.getElementById('dash-macro-carbs').textContent = `${consumedCarbs}g`;
        document.getElementById('dash-macro-carbs-pct').textContent = `Carbs · ${cPct}%`;

        document.getElementById('dash-macro-fat').textContent = `${consumedFat}g`;
        document.getElementById('dash-macro-fat-pct').textContent = `Fat · ${fPct}%`;

        // Render Water Tracker
        renderWaterTracker(goals.water || 2.5);

        // Render Today's Meals list
        const mealListEl = document.getElementById('dash-meal-list');
        if (mealListEl) {
            if (todaysMeals.length === 0) {
                mealListEl.innerHTML = `
                    <div style="padding: 24px; text-align: center; color: var(--muted);">
                        <i data-lucide="coffee" class="icon-lg" style="margin-bottom: 8px;"></i>
                        <div>No meals logged yet today.</div>
                        <span class="tiny">Scan a plate or add a meal to begin tracking!</span>
                    </div>
                `;
            } else {
                mealListEl.innerHTML = todaysMeals.map(m => `
                    <div class="meal" data-meal-id="${m.id}" title="Click to view meal details">
                        <img class="thumb" src="${escapeHTML(m.image || 'assets/nourish-bowl.png')}" alt="${escapeHTML(m.name)}" onerror="this.src='assets/nourish-bowl.png'">
                        <div class="meal-info">
                            <div class="meal-name">${escapeHTML(m.name)}</div>
                            <div class="meal-time">
                                <i data-lucide="clock" class="icon-sm"></i>
                                <span>${escapeHTML(m.type)} · ${escapeHTML(m.time || '12:00 PM')}</span>
                            </div>
                            <div class="meal-macros-inline">
                                <span>P: ${m.protein || 0}g</span>
                                <span>•</span>
                                <span>C: ${m.carbs || 0}g</span>
                                <span>•</span>
                                <span>F: ${m.fat || 0}g</span>
                            </div>
                        </div>
                        <div class="meal-kcal">
                            ${m.calories || 0}<small>kcal</small>
                        </div>
                        <div class="meal-actions" onclick="event.stopPropagation();">
                            <button class="del-meal-btn" data-delete-id="${m.id}" title="Delete meal">
                                <i data-lucide="trash-2" class="icon-sm"></i>
                            </button>
                        </div>
                    </div>
                `).join('');
            }
        }

        // Render Week-at-a-Glance 7-day Bar Chart
        renderWeekChart(meals, targetKcal);
    }

    function renderWaterTracker(targetLiters) {
        const waterData = Storage.getWater();
        const targetMl = Math.round(targetLiters * 1000);
        const currentMl = waterData.ml || 0;
        const waterPct = Math.min(100, Math.round((currentMl / targetMl) * 100));

        const displayEl = document.getElementById('dash-water-display');
        const subtitleEl = document.getElementById('dash-water-subtitle');

        if (displayEl) {
            displayEl.textContent = `${currentMl.toLocaleString()} / ${targetMl.toLocaleString()} mL`;
        }
        if (subtitleEl) {
            subtitleEl.textContent = `Hydration Progress · ${waterPct}% of daily target`;
        }
    }

    function initWaterEvents() {
        const goals = Storage.getGoals();

        document.getElementById('btn-water-add-250')?.addEventListener('click', () => {
            const current = Storage.getWater().ml;
            Storage.saveWater(current + 250);
            renderWaterTracker(goals.water || 2.5);
            showToast('+250 mL logged!');
        });

        document.getElementById('btn-water-add-500')?.addEventListener('click', () => {
            const current = Storage.getWater().ml;
            Storage.saveWater(current + 500);
            renderWaterTracker(goals.water || 2.5);
            showToast('+500 mL water logged!');
        });

        document.getElementById('btn-water-sub-250')?.addEventListener('click', () => {
            const current = Storage.getWater().ml;
            Storage.saveWater(Math.max(0, current - 250));
            renderWaterTracker(goals.water || 2.5);
            showToast('-250 mL water removed');
        });

        document.getElementById('btn-water-reset')?.addEventListener('click', () => {
            if (confirm("Reset today's water intake to 0 mL?")) {
                Storage.saveWater(0);
                renderWaterTracker(goals.water || 2.5);
                showToast("Today's water reset to 0 mL");
            }
        });
    }

    function renderWeekChart(meals, targetKcal) {
        const chartEl = document.getElementById('dash-week-chart');
        if (!chartEl) return;

        const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

        const daysData = [];
        for (let i = 6; i >= 0; i--) {
            const d = new Date();
            d.setDate(d.getDate() - i);
            const dateStr = d.toISOString().split('T')[0];
            const dayName = dayNames[d.getDay()];

            const dayTotal = meals
                .filter(m => m.date === dateStr)
                .reduce((acc, m) => acc + (Number(m.calories) || 0), 0);

            daysData.push({
                name: dayName,
                date: dateStr,
                kcal: dayTotal,
                isToday: i === 0
            });
        }

        const maxKcal = Math.max(targetKcal * 1.15, ...daysData.map(d => d.kcal), 2200);

        chartEl.innerHTML = daysData.map(d => {
            const heightPct = Math.max(8, Math.min(100, Math.round((d.kcal / maxKcal) * 100)));
            return `
                <div class="barcol">
                    <div class="bar ${d.isToday ? 'hi' : ''}" style="height: ${heightPct}%;">
                        <span class="bar-val">${d.kcal} kcal</span>
                    </div>
                    <span class="day">${d.name}</span>
                </div>
            `;
        }).join('');
    }

    // -------------------------------------------------------------
    // SCREEN 2: AUTHENTIC DUAL-VIEW FOOD RECOGNITION & INTERACTIVE ENGINE
    // -------------------------------------------------------------
    window.currentAnalysisData = null;
    window.currentIngredients = {};
    window.currentDish = "Plated Meal";

    function getFoodEmoji(name) {
        const n = (name || '').toLowerCase();
        if (n.includes('egg') || n.includes('omelet')) return '🍳';
        if (n.includes('toast') || n.includes('bread') || n.includes('sourdough')) return '🍞';
        if (n.includes('avocado')) return '🥑';
        if (n.includes('rice') || n.includes('quinoa') || n.includes('grain')) return '🍚';
        if (n.includes('chicken') || n.includes('poultry') || n.includes('meat') || n.includes('turkey')) return '🍗';
        if (n.includes('beef') || n.includes('steak')) return '🥩';
        if (n.includes('fish') || n.includes('salmon') || n.includes('tuna')) return '🐟';
        if (n.includes('salad') || n.includes('spinach') || n.includes('kale') || n.includes('lettuce')) return '🥗';
        if (n.includes('tofu') || n.includes('bean') || n.includes('chickpea') || n.includes('dal') || n.includes('lentil')) return '🧆';
        if (n.includes('cheese') || n.includes('paneer')) return '🧀';
        if (n.includes('potato') || n.includes('fries')) return '🥔';
        if (n.includes('apple') || n.includes('fruit') || n.includes('berry')) return '🍎';
        if (n.includes('pasta') || n.includes('noodle')) return '🍝';
        if (n.includes('soup') || n.includes('broth') || n.includes('curry')) return '🍲';
        return '🍽️';
    }

    function renderScanner() {
        const meals = Storage.getMeals();
        const galleryEl = document.getElementById('scan-recent-gallery');
        if (galleryEl) {
            const recent = meals.slice(0, 4);
            if (recent.length === 0) {
                galleryEl.innerHTML = `<div class="tiny" style="grid-column: span 4; color: var(--muted);">No scans recorded yet.</div>`;
            } else {
                galleryEl.innerHTML = recent.map(m => `
                    <div class="card" style="padding: 12px; cursor: pointer;" onclick="window.nutriScanSelectRecentMeal('${m.id}')" title="Click to load into scanner">
                        <img src="${escapeHTML(m.image || 'assets/nourish-bowl.png')}" style="width: 100%; height: 95px; object-fit: cover; border-radius: 12px;" onerror="this.src='assets/nourish-bowl.png'">
                        <b style="font-size: 12px; display: block; margin: 8px 0 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${escapeHTML(m.name)}</b>
                        <span class="tiny">${m.calories} kcal · ${escapeHTML(m.type)}</span>
                    </div>
                `).join('');
            }
        }
    }

    function validateImageFile(file) {
        if (!file) return { valid: false, message: "No file selected." };
        if (file.size === 0) return { valid: false, message: "The selected file is empty (0 bytes)." };
        const validExts = /\.(jpe?g|png|webp|heic|heif|bmp)$/i;
        if ((file.type && !file.type.startsWith("image/")) || (!file.type && !validExts.test(file.name))) {
            return { valid: false, message: "Please upload a valid image file (JPEG, PNG, WebP)." };
        }
        return { valid: true };
    }

    function setPreviewImage(file, previewEl, placeholderEl, isTop = true) {
        if (!file) return;
        const check = validateImageFile(file);
        if (!check.valid) {
            showToast(`⚠️ ${check.message}`, true);
            return;
        }
        if (isTop) {
            uploadedTopFile = file;
            clearPlateSpatialOverlay();
        } else {
            uploadedSideFile = file;
        }
        const url = URL.createObjectURL(file);
        previewEl.src = url;
        previewEl.style.display = "block";
        if (placeholderEl) placeholderEl.style.display = "none";
        const previewContainer = previewEl.closest(".upload-preview, .preview-area");
        if (previewContainer) previewContainer.classList.add("has-image");
    }

    function setupDragAndDrop(panel, input, previewEl, placeholderEl, isTop = true) {
        if (!panel) return;
        ["dragenter", "dragover"].forEach(evt => {
            panel.addEventListener(evt, (e) => {
                e.preventDefault();
                e.stopPropagation();
                panel.classList.add("drag-over");
            });
        });
        ["dragleave", "drop"].forEach(evt => {
            panel.addEventListener(evt, (e) => {
                e.preventDefault();
                e.stopPropagation();
                panel.classList.remove("drag-over");
            });
        });
        panel.addEventListener("drop", (e) => {
            if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                const file = e.dataTransfer.files[0];
                const check = validateImageFile(file);
                if (!check.valid) {
                    showToast(`⚠️ ${check.message}`, true);
                    return;
                }
                if (input) input.files = e.dataTransfer.files;
                setPreviewImage(file, previewEl, placeholderEl, isTop);
                showToast(`${isTop ? "Overhead" : "Eye-level"} photo added!`);
            }
        });
    }

    function showSideView() {
        const sidePanel = document.getElementById("sideUploadPanel");
        const addBtn = document.getElementById("addSideViewBtn");
        const grid = document.getElementById("uploadGrid");
        if (sidePanel) sidePanel.style.display = "flex";
        if (addBtn) addBtn.style.display = "none";
        if (grid) grid.classList.add("has-side-view");
    }

    function hideSideView() {
        const sidePanel = document.getElementById("sideUploadPanel");
        const addBtn = document.getElementById("addSideViewBtn");
        const grid = document.getElementById("uploadGrid");
        const sideImageInput = document.getElementById("sideImage");
        const sideCameraInput = document.getElementById("sideCameraInput");
        const sidePreview = document.getElementById("sidePreview");
        const sidePlaceholder = document.getElementById("sidePlaceholder");

        if (sidePanel) sidePanel.style.display = "none";
        if (addBtn) addBtn.style.display = "inline-flex";
        if (grid) grid.classList.remove("has-side-view");

        if (sideImageInput) sideImageInput.value = "";
        if (sideCameraInput) sideCameraInput.value = "";
        uploadedSideFile = null;

        if (sidePreview) {
            sidePreview.src = "";
            sidePreview.style.display = "none";
            const sideCont = sidePreview.closest(".upload-preview, .preview-area");
            if (sideCont) sideCont.classList.remove("has-image");
        }
        if (sidePlaceholder) sidePlaceholder.style.display = "flex";
    }

    function clearUploads() {
        const topImageInput = document.getElementById("topImage");
        const topCameraInput = document.getElementById("topCameraInput");
        const topPreview = document.getElementById("topPreview");
        const topPlaceholder = document.getElementById("topPlaceholder");
        const resultDiv = document.getElementById("result");
        const hintInput = document.getElementById("scanFoodHint");

        if (topImageInput) topImageInput.value = "";
        if (topCameraInput) topCameraInput.value = "";
        uploadedTopFile = null;

        if (topPreview) {
            topPreview.src = "";
            topPreview.style.display = "none";
            const topCont = topPreview.closest(".upload-preview, .preview-area");
            if (topCont) topCont.classList.remove("has-image");
        }
        if (topPlaceholder) topPlaceholder.style.display = "flex";
        if (hintInput) hintInput.value = "";
        clearPlateSpatialOverlay();

        hideSideView();

        window.currentAnalysisData = null;
        window.currentIngredients = {};
        window.currentDish = "Plated Meal";

        if (resultDiv) {
            resultDiv.innerHTML = `
                <div class="empty-state card" style="text-align: center; padding: 36px 20px;">
                    <div class="empty-state-icon" style="font-size: 38px; margin-bottom: 8px;">🍽️</div>
                    <h3 class="empty-state-heading" style="font-size: 16px; font-weight: 800; color: var(--ink);">Photos cleared</h3>
                    <p class="empty-state-body tiny" style="margin-top: 4px;">Upload an overhead photo of your plate, then click Analyze.</p>
                </div>
            `;
        }
        showToast("Scanner inputs cleared");
    }

    window.showSideView = showSideView;
    window.hideSideView = hideSideView;
    window.clearUploads = clearUploads;

    window.retryScannerUpload = function () {
        clearUploads();
        const topUploadPanel = document.getElementById("topUploadPanel");
        const topImageInput = document.getElementById("topImage");
        if (topUploadPanel) {
            topUploadPanel.scrollIntoView({ behavior: "smooth", block: "center" });
        }
        if (topImageInput) {
            topImageInput.click();
        }
    };
    window.clearUploads = clearUploads;
    window.nutriScanSwitchView = switchView;

    function initScannerUploads() {
        const topImageInput = document.getElementById("topImage");
        const topCameraInput = document.getElementById("topCameraInput");
        const topPreview = document.getElementById("topPreview");
        const topPlaceholder = document.getElementById("topPlaceholder");
        const topUploadPanel = document.getElementById("topUploadPanel");

        const sideImageInput = document.getElementById("sideImage");
        const sideCameraInput = document.getElementById("sideCameraInput");
        const sidePreview = document.getElementById("sidePreview");
        const sidePlaceholder = document.getElementById("sidePlaceholder");
        const sideUploadPanel = document.getElementById("sideUploadPanel");

        if (topImageInput) {
            topImageInput.addEventListener("change", function () {
                if (topImageInput.files && topImageInput.files.length > 0) {
                    setPreviewImage(topImageInput.files[0], topPreview, topPlaceholder, true);
                }
            });
        }

        if (topCameraInput) {
            topCameraInput.addEventListener("change", function () {
                if (topCameraInput.files && topCameraInput.files.length > 0) {
                    setPreviewImage(topCameraInput.files[0], topPreview, topPlaceholder, true);
                }
            });
        }

        if (sideImageInput) {
            sideImageInput.addEventListener("change", function () {
                if (sideImageInput.files && sideImageInput.files.length > 0) {
                    setPreviewImage(sideImageInput.files[0], sidePreview, sidePlaceholder, false);
                }
            });
        }

        if (sideCameraInput) {
            sideCameraInput.addEventListener("change", function () {
                if (sideCameraInput.files && sideCameraInput.files.length > 0) {
                    setPreviewImage(sideCameraInput.files[0], sidePreview, sidePlaceholder, false);
                }
            });
        }

        setupDragAndDrop(topUploadPanel, topImageInput, topPreview, topPlaceholder, true);
        setupDragAndDrop(sideUploadPanel, sideImageInput, sidePreview, sidePlaceholder, false);

        // Example plate presets
        const btnToast = document.getElementById("btn-sample-toast");
        if (btnToast) {
            btnToast.addEventListener("click", () => loadSampleDish("toast"));
        }

        const btnBowl = document.getElementById("btn-sample-bowl");
        if (btnBowl) {
            btnBowl.addEventListener("click", () => loadSampleDish("bowl"));
        }

        initCamera();
    }

    async function loadSampleDish(type) {
        const isToast = type === "toast";
        const imgPath = isToast ? "assets/avocado-egg-toast.png" : "assets/nourish-bowl.png";
        const dishTitle = isToast ? "Avocado egg toast" : "Garden nourish bowl";
        const hintText = isToast ? "Avocado egg toast with poached egg" : "Garden nourish bowl with quinoa & chickpeas";

        const topPreview = document.getElementById("topPreview");
        const topPlaceholder = document.getElementById("topPlaceholder");
        const scanFoodHint = document.getElementById("scanFoodHint");

        if (topPreview) {
            topPreview.src = imgPath;
            topPreview.style.display = "block";
        }
        if (topPlaceholder) topPlaceholder.style.display = "none";
        const previewCont = topPreview?.closest(".upload-preview");
        if (previewCont) previewCont.classList.add("has-image");
        if (scanFoodHint) scanFoodHint.value = hintText;

        try {
            const resp = await fetch(imgPath);
            const blob = await resp.blob();
            uploadedTopFile = new File([blob], isToast ? "avocado_toast.png" : "nourish_bowl.png", { type: "image/png" });
        } catch (e) {
            console.warn("Sample blob fetch note:", e);
        }

        if (isToast) {
            window.currentDish = "Avocado egg toast";
            window.currentIngredients = {
                "Whole grain sourdough": 70,
                "Poached egg": 90,
                "Fresh avocado": 70
            };
            window.currentAnalysisData = {
                dish: "Avocado egg toast",
                scale_reference: { detected: true, plate_diameter_cm: 25.0 },
                ingredients: { ...window.currentIngredients },
                nutrition: [
                    { ingredient: "Whole grain sourdough", weight_g: 70, energy_kcal: 165, carb_g: 31, protein_g: 5, fat_g: 2, fibre_g: 3 },
                    { ingredient: "Poached egg", weight_g: 90, energy_kcal: 135, carb_g: 1, protein_g: 11, fat_g: 10, fibre_g: 0 },
                    { ingredient: "Fresh avocado", weight_g: 70, energy_kcal: 120, carb_g: 6, protein_g: 2, fat_g: 10, fibre_g: 4 }
                ],
                recipe_totals: {
                    total_weight_g: 230,
                    energy_kcal: 420,
                    carb_g: 38,
                    protein_g: 18,
                    fat_g: 22,
                    fibre_g: 7
                }
            };
        } else {
            window.currentDish = "Garden nourish bowl";
            window.currentIngredients = {
                "Steamed quinoa & brown rice": 150,
                "Crispy spiced chickpeas & tofu": 120,
                "Roasted greens & tahini": 100
            };
            window.currentAnalysisData = {
                dish: "Garden nourish bowl",
                scale_reference: { detected: true, plate_diameter_cm: 26.0 },
                ingredients: { ...window.currentIngredients },
                nutrition: [
                    { ingredient: "Steamed quinoa & brown rice", weight_g: 150, energy_kcal: 210, carb_g: 42, protein_g: 6, fat_g: 3, fibre_g: 4 },
                    { ingredient: "Crispy spiced chickpeas & tofu", weight_g: 120, energy_kcal: 190, carb_g: 20, protein_g: 14, fat_g: 7, fibre_g: 5 },
                    { ingredient: "Roasted greens & tahini", weight_g: 100, energy_kcal: 120, carb_g: 14, protein_g: 4, fat_g: 8, fibre_g: 5 }
                ],
                recipe_totals: {
                    total_weight_g: 370,
                    energy_kcal: 520,
                    carb_g: 76,
                    protein_g: 24,
                    fat_g: 18,
                    fibre_g: 14
                }
            };
        }

        renderAnalysisResults(window.currentAnalysisData);
        showToast(`Loaded ${dishTitle} sample! Adjust portions below or click Analyze.`);
    }

    function initCamera() {
        const btnCloseCam = document.getElementById("btn-close-camera");
        const btnFlipCam = document.getElementById("btn-camera-flip");
        const btnShutter = document.getElementById("btn-shutter-capture");
        const video = document.getElementById("camera-feed");
        const canvas = document.getElementById("camera-canvas");

        if (btnCloseCam) {
            btnCloseCam.addEventListener("click", () => {
                stopCameraStream();
                closeModal("modal-camera");
            });
        }

        if (btnFlipCam) {
            btnFlipCam.addEventListener("click", async () => {
                cameraFacingMode = (cameraFacingMode === "environment") ? "user" : "environment";
                await startCameraStream();
            });
        }

        if (btnShutter) {
            btnShutter.addEventListener("click", () => {
                if (!video || !video.videoWidth || !video.videoHeight) {
                    showToast("Waiting for camera feed...", true);
                    return;
                }

                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext("2d");
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                canvas.toBlob(blob => {
                    if (blob) {
                        const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
                        const topPreview = document.getElementById("topPreview");
                        const topPlaceholder = document.getElementById("topPlaceholder");
                        setPreviewImage(file, topPreview, topPlaceholder, true);
                        showToast("Overhead photo captured from camera!");
                    }
                }, "image/jpeg", 0.92);

                stopCameraStream();
                closeModal("modal-camera");
            });
        }
    }

    async function startCameraStream() {
        stopCameraStream();
        const video = document.getElementById("camera-feed");
        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                showToast("Live camera not supported by this browser. Use file upload.", true);
                return;
            }

            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: cameraFacingMode,
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            });

            if (video) {
                video.srcObject = cameraStream;
            }
        } catch (err) {
            console.error("Camera access error:", err);
            showToast("Camera access denied or unavailable. Use file upload.", true);
        }
    }

    function stopCameraStream() {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
        const video = document.getElementById("camera-feed");
        if (video) video.srcObject = null;
    }

    // -------------------------------------------------------------
    // DUAL-VIEW INFERENCE ORCHESTRATION & 4-STAGE PIPELINE STEPPER
    // -------------------------------------------------------------
    async function analyzeFood() {
        const topImageInput = document.getElementById("topImage");
        const topCameraInput = document.getElementById("topCameraInput");
        const topPreview = document.getElementById("topPreview");
        const sideImageInput = document.getElementById("sideImage");
        const sideCameraInput = document.getElementById("sideCameraInput");
        const plateDiameterInput = document.getElementById("plateDiameter");
        const scanFoodHintInput = document.getElementById("scanFoodHint");
        const analyzeButton = document.getElementById("analyzeButton");
        const resultDiv = document.getElementById("result");

        let topFile = uploadedTopFile ||
            (topImageInput && topImageInput.files && topImageInput.files.length > 0 ? topImageInput.files[0] : null) ||
            (topCameraInput && topCameraInput.files && topCameraInput.files.length > 0 ? topCameraInput.files[0] : null);

        // If user selected sample or captured image without direct file object, convert preview src
        if (!topFile && topPreview && topPreview.src && topPreview.style.display !== "none" && !topPreview.src.startsWith("blob:null")) {
            try {
                const resp = await fetch(topPreview.src);
                const blob = await resp.blob();
                topFile = new File([blob], "overhead_plate.png", { type: blob.type || "image/png" });
                uploadedTopFile = topFile;
            } catch (e) {
                console.warn("Could not convert preview to blob:", e);
            }
        }

        if (!topFile) {
            if (resultDiv) {
                resultDiv.innerHTML = `
                    <div class="alert-error" style="background: rgba(220, 38, 38, 0.08); border: 1px solid #dc2626; border-radius: var(--r-md); padding: 16px; margin: 16px 0; color: #991b1b;">
                        <strong style="display:block; margin-bottom: 4px;">Missing overhead photo.</strong>
                        Please upload or capture an overhead photo of your plate, then click Analyze.
                    </div>
                `;
                resultDiv.scrollIntoView({ behavior: "smooth" });
            }
            return;
        }

        const sideFile = uploadedSideFile ||
            (sideImageInput && sideImageInput.files && sideImageInput.files.length > 0 ? sideImageInput.files[0] : null) ||
            (sideCameraInput && sideCameraInput.files && sideCameraInput.files.length > 0 ? sideCameraInput.files[0] : null);

        const plateDiameter = plateDiameterInput ? parseFloat(plateDiameterInput.value) || 25.0 : 25.0;
        const foodHint = scanFoodHintInput ? scanFoodHintInput.value.trim() : "";

        // Construct clean FormData on demand to prevent body stream reuse or abortion lockouts
        function buildFormData() {
            const fd = new FormData();
            fd.append("top_image", topFile);
            if (sideFile) fd.append("side_image", sideFile);
            fd.append("plate_diameter_cm", plateDiameter.toString());
            if (foodHint) fd.append("food_hint", foodHint);
            return fd;
        }

        if (analyzeButton) {
            analyzeButton.disabled = true;
            analyzeButton.innerHTML = `<i data-lucide="loader-2" class="icon-sm"></i> <span>Analyzing meal…</span>`;
            renderLucide();
        }

        let _elapsed = 0;
        const steps = [
            { label: "Validating images & calibrating plate reference scale", pct: 20 },
            { label: "Detecting plate boundary & food layout", pct: 45 },
            { label: "Identifying meal components & portion depth", pct: 75 },
            { label: "Calculating portion weights & nutritional breakdown", pct: 95 }
        ];

        function renderStepperSkeleton() {
            const stepListHtml = steps.map((s, i) => `
                <div class="step-item ${i === 0 ? 'active' : ''}" id="stepItem${i}">
                    <div class="step-bullet" id="stepBullet${i}">${i + 1}</div>
                    <span>${s.label}</span>
                </div>
            `).join('');
            return `
                <div class="pipeline-progress-box">
                    <div class="stepper-header">
                        <span class="stepper-title">Analyzing meal with NutriScanAI</span>
                        <span class="stepper-timer" id="stepperTimer">0s</span>
                    </div>
                    <div class="progress-bar-track">
                        <div class="progress-bar-fill" id="stepperFill" style="width: 15%"></div>
                    </div>
                    <div class="stepper-list">${stepListHtml}</div>
                </div>`;
        }

        function updateStepper(sec) {
            let activeIdx = 0;
            if (sec >= 3 && sec < 8) activeIdx = 1;
            else if (sec >= 8 && sec < 20) activeIdx = 2;
            else if (sec >= 20) activeIdx = 3;

            const pct = Math.min(95, Math.max(15, steps[activeIdx].pct + Math.floor((sec % 5) * 2)));

            const timerEl = document.getElementById("stepperTimer");
            if (timerEl) timerEl.textContent = `${sec}s`;

            const fillEl = document.getElementById("stepperFill");
            if (fillEl) fillEl.style.width = `${pct}%`;

            steps.forEach((_, idx) => {
                const itemEl = document.getElementById(`stepItem${idx}`);
                const bulletEl = document.getElementById(`stepBullet${idx}`);
                if (!itemEl || !bulletEl) return;
                itemEl.className = idx === activeIdx ? "step-item active" : (idx < activeIdx ? "step-item done" : "step-item");
                bulletEl.textContent = idx < activeIdx ? "✓" : String(idx + 1);
            });
        }

        resultDiv.innerHTML = renderStepperSkeleton();
        resultDiv.scrollIntoView({ behavior: "smooth" });

        const _timerInterval = setInterval(() => {
            _elapsed++;
            updateStepper(_elapsed);
        }, 1000);

        try {
            let combinedData = null;

            // Strategy A: Try unified dual analysis endpoint on port 8000
            try {
                const isPort8000 = (typeof window !== 'undefined' && window.location && window.location.port === "8000");
                const dualUrl = isPort8000 ? "/analyze/dual" : `${BACKEND_BASE_URL}/analyze/dual`;
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 120000);

                const resp = await fetch(dualUrl, {
                    method: "POST",
                    body: buildFormData(),
                    signal: controller.signal
                });
                clearTimeout(timeoutId);

                if (resp.ok) {
                    const data = await resp.json();
                    if (data && (data.nutrition || data.is_food === false)) {
                        combinedData = data;
                    }
                } else if (resp.status === 422) {
                    const errData = await resp.json().catch(() => ({}));
                    combinedData = {
                        is_food: false,
                        dish: "Non-Food Image",
                        reason: errData.detail || "Image unreadable or rejected by validation check",
                        message: errData.detail || "The uploaded image could not be recognized as food. Please upload a clear photo of your meal."
                    };
                }
            } catch (errDual) {
                console.warn("/analyze/dual unavailable, attempting separate Vision + ML calls:", errDual);
            }

            // Strategy B: If unified failed, try Vision (8001) then ML (8000)
            if (!combinedData) {
                try {
                    const visionUrl = window.VISION_API_URL || `${VISION_BASE_URL}/vision`;
                    const vController = new AbortController();
                    const vTimeout = setTimeout(() => vController.abort(), 120000);

                    const vResp = await fetch(visionUrl, { method: "POST", body: buildFormData(), signal: vController.signal });
                    clearTimeout(vTimeout);

                    if (vResp.ok) {
                        const visionData = await vResp.json();
                        if (visionData.is_food === false) {
                            combinedData = visionData;
                        } else {
                            const mlPayload = {
                                dish: visionData.dish || foodHint || null,
                                ingredients: visionData.ingredients || {},
                                estimates: visionData.estimates || []
                            };

                            try {
                                const isPort8000 = (typeof window !== 'undefined' && window.location && window.location.port === "8000");
                                const nutritionUrl = window.NUTRITION_API_URL || (isPort8000 ? "/predict/vision" : `${BACKEND_BASE_URL}/predict/vision`);
                                const mlResp = await fetch(nutritionUrl, {
                                    method: "POST",
                                    headers: { "Content-Type": "application/json" },
                                    body: JSON.stringify(mlPayload),
                                });

                                if (mlResp.ok) {
                                    const mlData = await mlResp.json();
                                    combinedData = {
                                        ...visionData,
                                        nutrition: mlData.nutrition,
                                        recipe_totals: mlData.recipe_totals,
                                    };
                                } else {
                                    combinedData = visionData;
                                }
                            } catch (errML) {
                                console.warn("Prediction endpoint /predict/vision unavailable:", errML);
                                combinedData = visionData;
                            }
                        }
                    } else if (vResp.status === 422) {
                        const errData = await vResp.json().catch(() => ({}));
                        combinedData = {
                            is_food: false,
                            dish: "Non-Food Image",
                            reason: errData.detail || "Image unreadable or rejected by validation check",
                            message: errData.detail || "The uploaded image could not be recognized as food. Please upload a clear photo of your meal."
                        };
                    }
                } catch (errVision) {
                    console.warn("Vision endpoint /vision unavailable:", errVision);
                }
            }

            clearInterval(_timerInterval);

            if (combinedData) {
                // Non-food edge case check
                const isNonFood = combinedData.is_food === false ||
                    (Array.isArray(combinedData.estimates) && combinedData.estimates.length === 0 &&
                     (!combinedData.nutrition || combinedData.nutrition.length === 0) &&
                     (!combinedData.ingredients || Object.keys(combinedData.ingredients).length === 0));

                if (isNonFood) {
                    window.currentAnalysisData = combinedData;
                    renderNonFoodDetected(combinedData);
                    showToast("⚠️ No food detected. Please upload a clear photo of your meal.", true);
                    return;
                }

                window.currentDish = combinedData.dish || foodHint || "Plated Meal";
                window.currentIngredients = { ...(combinedData.ingredients || {}) };

                if (Object.keys(window.currentIngredients).length === 0 && Array.isArray(combinedData.nutrition)) {
                    combinedData.nutrition.forEach(item => {
                        window.currentIngredients[item.ingredient] = item.weight_g;
                    });
                } else if (Object.keys(window.currentIngredients).length === 0 && Array.isArray(combinedData.estimates)) {
                    combinedData.estimates.forEach(item => {
                        const w = (typeof item.weight_g === 'object') ? (item.weight_g.estimate || 100) : (item.weight_g || 100);
                        window.currentIngredients[item.name] = Math.max(30, Math.round(w));
                    });
                }

                // If nutritional info is zero or missing (e.g. plate rim cropped/uncalibrated), compute baseline nutrition
                const totals = combinedData.recipe_totals || {};
                if (!totals.energy_kcal || totals.energy_kcal === 0) {
                    if (Object.keys(window.currentIngredients).length > 0) {
                        let totalW = 0, totalKcal = 0, totalCarb = 0, totalProt = 0, totalFat = 0, totalFib = 0;
                        const fallbackNutrition = [];
                        for (const [name, weight] of Object.entries(window.currentIngredients)) {
                            const w = Math.max(30, parseFloat(weight) || 100);
                            window.currentIngredients[name] = Math.round(w);
                            totalW += w;
                            const factor = w / 100.0;
                            const n = name.toLowerCase();
                            let pk = 140, pc = 18, pp = 6, pf = 5, pfb = 2;
                            if (n.includes('ghee') || n.includes('oil') || n.includes('butter')) {
                                pk = 900; pc = 0; pp = 0; pf = 100; pfb = 0;
                            } else if (n.includes('water')) {
                                pk = 0; pc = 0; pp = 0; pf = 0; pfb = 0;
                            } else if (n.includes('egg')) {
                                pk = 150; pc = 1; pp = 13; pf = 11; pfb = 0;
                            } else if (n.includes('bread') || n.includes('roti') || n.includes('naan')) {
                                pk = 240; pc = 45; pp = 8; pf = 3; pfb = 4;
                            } else if (n.includes('rice') || n.includes('grain') || n.includes('pasta')) {
                                pk = 130; pc = 28; pp = 3; pf = 1; pfb = 1;
                            } else if (n.includes('chicken') || n.includes('meat') || n.includes('fish') || n.includes('paneer')) {
                                pk = 180; pc = 2; pp = 24; pf = 9; pfb = 0;
                            } else if (n.includes('dal') || n.includes('curry') || n.includes('sambar') || n.includes('chole')) {
                                pk = 120; pc = 16; pp = 7; pf = 4; pfb = 4;
                            } else if (n.includes('salad') || n.includes('vegetable') || n.includes('greens')) {
                                pk = 45; pc = 8; pp = 2; pf = 0.5; pfb = 3;
                            }
                            const itemKcal = Math.round(pk * factor);
                            const itemCarb = Math.round(pc * factor * 10) / 10;
                            const itemProt = Math.round(pp * factor * 10) / 10;
                            const itemFat = Math.round(pf * factor * 10) / 10;
                            const itemFib = Math.round(pfb * factor * 10) / 10;
                            totalKcal += itemKcal; totalCarb += itemCarb; totalProt += itemProt; totalFat += itemFat; totalFib += itemFib;
                            fallbackNutrition.push({
                                ingredient: name,
                                weight_g: Math.round(w),
                                energy_kcal: itemKcal,
                                carb_g: itemCarb,
                                protein_g: itemProt,
                                fat_g: itemFat,
                                fibre_g: itemFib
                            });
                        }
                        combinedData.nutrition = fallbackNutrition;
                        combinedData.recipe_totals = {
                            total_weight_g: Math.round(totalW),
                            energy_kcal: Math.round(totalKcal),
                            carb_g: Math.round(totalCarb),
                            protein_g: Math.round(totalProt),
                            fat_g: Math.round(totalFat),
                            fibre_g: Math.round(totalFib)
                        };
                    }
                }

                window.currentAnalysisData = combinedData;
                renderAnalysisResults(window.currentAnalysisData);
                showToast("✦ Food recognition complete!");
            } else {
                // If neither endpoint responded (backend offline), offer intelligent fallback & demo
                resultDiv.innerHTML = `
                    <div class="alert-error" style="background: rgba(239, 68, 68, 0.08); border: 1px solid #ef4444; border-radius: var(--r-md); padding: 18px 20px; margin: 16px 0; color: #991b1b;">
                        <strong style="display:block; margin-bottom: 6px; font-size: 15px;">Backend Service Offline (Ports 8000 / 8001)</strong>
                        <p style="font-size: 13px; line-height: 1.5; margin-bottom: 12px;">The Vision service (port 8001) and ML service (port 8000) are not actively responding. Start them via PowerShell (<code>.\\start_services.ps1</code>) or run standalone demo mode below.</p>
                        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                            <button type="button" class="btn btn-sm btn-outline" onclick="window.loadDemoAnalysis()" style="font-size: 12px; background: white;">
                                ⚡ Run interactive demo analysis for this plate
                            </button>
                            <button type="button" class="btn btn-sm btn-forest" onclick="analyzeFood()" style="font-size: 12px;">
                                ↻ Retry connection
                            </button>
                        </div>
                    </div>
                `;
            }

        } catch (error) {
            clearInterval(_timerInterval);
            console.error("API error:", error);
            resultDiv.innerHTML = `
                <div class="alert-error" style="background: rgba(239, 68, 68, 0.08); border: 1px solid #ef4444; border-radius: var(--r-md); padding: 18px 20px; margin: 16px 0; color: #991b1b;">
                    <strong style="display:block; margin-bottom: 6px; font-size: 15px;">Connection or Processing Notice</strong>
                    <p style="font-size: 13px; line-height: 1.5; margin-bottom: 12px;">${escapeHTML(error.message || "Failed to reach vision/ML backend")}. Ensure the services are running on ports 8000 and 8001.</p>
                    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                        <button type="button" class="btn btn-sm btn-outline" onclick="window.loadDemoAnalysis()" style="font-size: 12px; background: white;">
                            ⚡ Run interactive demo analysis for this plate
                        </button>
                        <button type="button" class="btn btn-sm btn-forest" onclick="analyzeFood()" style="font-size: 12px;">
                            ↻ Retry analysis
                        </button>
                    </div>
                </div>
            `;
        } finally {
            if (analyzeButton) {
                analyzeButton.disabled = false;
                analyzeButton.innerHTML = `<i data-lucide="scan"></i> <span>Analyze meal</span>`;
                renderLucide();
            }
        }
    }

    window.analyzeFood = analyzeFood;

    // -------------------------------------------------------------
    // HUMAN-IN-THE-LOOP INTERACTIVE TRAY & RE-CALCULATION ENGINE
    // -------------------------------------------------------------
    async function recalculateNutrition() {
        if (!window.currentIngredients || Object.keys(window.currentIngredients).length === 0) {
            showToast("Please add at least one ingredient to calculate nutrition.", true);
            return;
        }

        const recalcBtn = document.getElementById("recalcBtn");
        if (recalcBtn) {
            recalcBtn.disabled = true;
            recalcBtn.innerHTML = '<i data-lucide="loader-2" class="icon-sm"></i> <span>Recalculating…</span>';
            renderLucide();
        }

        const mlPayload = {
            dish: window.currentDish || null,
            ingredients: window.currentIngredients,
        };

        let calculated = false;

        // 1. Attempt CatBoost prediction backend on port 8000
        try {
            const isPort8000 = (typeof window !== 'undefined' && window.location && window.location.port === "8000");
            const nutritionUrl = window.NUTRITION_API_URL || (isPort8000 ? "/predict/vision" : `${BACKEND_BASE_URL}/predict/vision`);
            const resp = await fetch(nutritionUrl, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(mlPayload),
            });

            if (resp.ok) {
                const mlData = await resp.json();
                if (mlData && mlData.nutrition && mlData.recipe_totals) {
                    window.currentAnalysisData.nutrition = mlData.nutrition;
                    window.currentAnalysisData.recipe_totals = mlData.recipe_totals;
                    renderAnalysisResults(window.currentAnalysisData);
                    showToast("✦ CatBoost regression recalculated!");
                    calculated = true;
                }
            }
        } catch (err) {
            console.warn("Backend recalculation unavailable, using local nutritional estimation engine:", err);
        }

        // 2. Intelligent local calculation fallback (guarantees UI never freezes or fails)
        if (!calculated) {
            let totalW = 0, totalKcal = 0, totalCarb = 0, totalProt = 0, totalFat = 0, totalFib = 0;
            const newNutrition = [];

            for (const [name, weight] of Object.entries(window.currentIngredients)) {
                const w = parseFloat(weight) || 50;
                totalW += w;
                const factor = w / 100.0;
                const n = name.toLowerCase();

                let per100_kcal = 140, per100_carb = 18, per100_prot = 6, per100_fat = 5, per100_fib = 2;

                if (n.includes('egg') || n.includes('omelet')) {
                    per100_kcal = 150; per100_carb = 1; per100_prot = 13; per100_fat = 11; per100_fib = 0;
                } else if (n.includes('bread') || n.includes('toast') || n.includes('sourdough') || n.includes('roti') || n.includes('naan')) {
                    per100_kcal = 240; per100_carb = 45; per100_prot = 8; per100_fat = 3; per100_fib = 4;
                } else if (n.includes('avocado')) {
                    per100_kcal = 160; per100_carb = 8; per100_prot = 2; per100_fat = 15; per100_fib = 7;
                } else if (n.includes('rice') || n.includes('quinoa') || n.includes('pasta') || n.includes('noodle')) {
                    per100_kcal = 130; per100_carb = 28; per100_prot = 3; per100_fat = 1; per100_fib = 2;
                } else if (n.includes('chicken') || n.includes('poultry') || n.includes('turkey')) {
                    per100_kcal = 165; per100_carb = 0; per100_prot = 31; per100_fat = 4; per100_fib = 0;
                } else if (n.includes('beef') || n.includes('steak') || n.includes('meat')) {
                    per100_kcal = 220; per100_carb = 0; per100_prot = 26; per100_fat = 13; per100_fib = 0;
                } else if (n.includes('fish') || n.includes('salmon') || n.includes('tuna')) {
                    per100_kcal = 180; per100_carb = 0; per100_prot = 24; per100_fat = 9; per100_fib = 0;
                } else if (n.includes('chickpea') || n.includes('tofu') || n.includes('bean') || n.includes('dal') || n.includes('lentil')) {
                    per100_kcal = 150; per100_carb = 18; per100_prot = 11; per100_fat = 5; per100_fib = 5;
                } else if (n.includes('salad') || n.includes('spinach') || n.includes('kale') || n.includes('vegetable') || n.includes('greens')) {
                    per100_kcal = 45; per100_carb = 6; per100_prot = 2; per100_fat = 1; per100_fib = 3;
                } else if (n.includes('cheese') || n.includes('paneer')) {
                    per100_kcal = 300; per100_carb = 3; per100_prot = 20; per100_fat = 24; per100_fib = 0;
                } else if (n.includes('fruit') || n.includes('berry') || n.includes('apple')) {
                    per100_kcal = 60; per100_carb = 14; per100_prot = 1; per100_fat = 0.5; per100_fib = 3;
                }

                const itemKcal = Math.round(per100_kcal * factor);
                const itemCarb = Math.round(per100_carb * factor);
                const itemProt = Math.round(per100_prot * factor);
                const itemFat  = Math.round(per100_fat * factor);
                const itemFib  = Math.round(per100_fib * factor);

                totalKcal += itemKcal;
                totalCarb += itemCarb;
                totalProt += itemProt;
                totalFat  += itemFat;
                totalFib  += itemFib;

                newNutrition.push({
                    ingredient: name,
                    weight_g: Math.round(w),
                    energy_kcal: itemKcal,
                    carb_g: itemCarb,
                    protein_g: itemProt,
                    fat_g: itemFat,
                    fibre_g: itemFib
                });
            }

            if (!window.currentAnalysisData) window.currentAnalysisData = {};
            window.currentAnalysisData.nutrition = newNutrition;
            window.currentAnalysisData.recipe_totals = {
                total_weight_g: Math.round(totalW),
                energy_kcal: Math.round(totalKcal),
                carb_g: Math.round(totalCarb),
                protein_g: Math.round(totalProt),
                fat_g: Math.round(totalFat),
                fibre_g: Math.round(totalFib)
            };
            renderAnalysisResults(window.currentAnalysisData);
            showToast("✦ Portions updated & nutrition recalculated!");
        }
    }

    function adjustWeight(name, delta) {
        if (!name) return;
        const targetName = (window.currentIngredients && window.currentIngredients[name] !== undefined)
            ? name
            : ((window.currentIngredients && window.currentIngredients[decodeURIComponent(name)] !== undefined) ? decodeURIComponent(name) : name);

        if (!window.currentIngredients || window.currentIngredients[targetName] === undefined) return;
        const current = window.currentIngredients[targetName];
        const updated = Math.max(10, current + delta);
        window.currentIngredients[targetName] = updated;

        const encName = encodeURIComponent(targetName);
        const input = document.getElementById(`weight_input_${encName}`) || document.getElementById(`weight_input_${targetName}`);
        if (input) input.value = updated;
    }

    function updateWeightManual(name, value) {
        if (!name) return;
        const targetName = (window.currentIngredients && window.currentIngredients[name] !== undefined)
            ? name
            : ((window.currentIngredients && window.currentIngredients[decodeURIComponent(name)] !== undefined) ? decodeURIComponent(name) : name);

        const val = parseFloat(value);
        if (!isNaN(val) && val > 0 && window.currentIngredients) {
            window.currentIngredients[targetName] = Math.round(val);
        }
    }

    function removeIngredient(name) {
        if (!name) return;
        const targetName = (window.currentIngredients && window.currentIngredients[name] !== undefined)
            ? name
            : ((window.currentIngredients && window.currentIngredients[decodeURIComponent(name)] !== undefined) ? decodeURIComponent(name) : name);

        if (window.currentIngredients) {
            delete window.currentIngredients[targetName];
        }
        if (window.currentAnalysisData && Array.isArray(window.currentAnalysisData.nutrition)) {
            window.currentAnalysisData.nutrition = window.currentAnalysisData.nutrition.filter(i => i.ingredient !== targetName && i.ingredient !== name);
        }
        recalculateNutrition();
        showToast(`Removed "${targetName}" from meal`);
    }

    function renameIngredient(name, nextName) {
        const targetName = (window.currentIngredients && window.currentIngredients[name] !== undefined)
            ? name
            : decodeURIComponent(name || "");
        const cleanName = (nextName || "").trim();
        if (!cleanName || !window.currentIngredients || !window.currentIngredients[targetName]) return;
        if (cleanName === targetName) return;
        const weight = window.currentIngredients[targetName];
        delete window.currentIngredients[targetName];
        window.currentIngredients[cleanName] = (window.currentIngredients[cleanName] || 0) + weight;
        recalculateNutrition();
        showToast(`Renamed "${targetName}" to "${cleanName}"`);
    }

    function addCustomIngredient() {
        const nameInput = document.getElementById("newIngrName");
        const weightInput = document.getElementById("newIngrWeight");
        if (!nameInput || !weightInput) return;

        const name = nameInput.value.trim();
        const weight = parseFloat(weightInput.value);

        if (!name) {
            showToast("Please enter an ingredient name.", true);
            nameInput.focus();
            return;
        }
        if (isNaN(weight) || weight <= 0) {
            showToast("Please enter a valid weight in grams.", true);
            weightInput.focus();
            return;
        }

        window.currentIngredients[name] = Math.round(weight);
        nameInput.value = "";
        weightInput.value = "100";
        recalculateNutrition();
    }

    window.adjustWeight = adjustWeight;
    window.updateWeightManual = updateWeightManual;
    window.removeIngredient = removeIngredient;
    window.renameIngredient = renameIngredient;
    window.addCustomIngredient = addCustomIngredient;
    window.recalculateNutrition = recalculateNutrition;

    window.loadDemoAnalysis = function () {
        const hint = (document.getElementById('scanFoodHint')?.value || '').trim();
        const plateDiam = parseFloat(document.getElementById('plateDiameter')?.value) || 25.0;
        const dishName = hint || (uploadedTopFile?.name?.replace(/\.[^/.]+$/, "") || "Plated Meal");

        const scaleFactor = Math.pow(plateDiam / 25.0, 1.8);
        const mainW = Math.round(180 * scaleFactor);
        const sideW = Math.round(110 * scaleFactor);
        const greenW = Math.round(80 * scaleFactor);

        window.currentDish = dishName;
        window.currentIngredients = {
            [`${dishName} (main)`]: mainW,
            "Protein component": sideW,
            "Fresh greens & sides": greenW
        };

        window.currentAnalysisData = {
            dish: dishName,
            scale_reference: {
                detected: true,
                plate_diameter_cm: plateDiam
            },
            ingredients: { ...window.currentIngredients },
            nutrition: [],
            recipe_totals: {}
        };

        recalculateNutrition();
        showToast(`Loaded interactive volumetric analysis for "${dishName}"!`);
    };

    // -------------------------------------------------------------
    // EDGE CASE: NON-FOOD OR UNRECOGNIZABLE IMAGE DETECTED
    // -------------------------------------------------------------
    function renderNonFoodDetected(data) {
        const resultDiv = document.getElementById("result");
        if (!resultDiv) return;

        const reason = escapeHTML((data && (data.reason || data.message)) || "No recognizable food items or plated meal found in image.");
        const hint = (document.getElementById("scanFoodHint")?.value || "").trim();

        resultDiv.innerHTML = `
            <div class="card edge-case-card" style="border: 1px solid #f59e0b; background: var(--paper-card); border-radius: var(--radius-lg); padding: 28px 24px; margin-top: 16px; box-shadow: var(--shadow-md);">
                <div style="display: flex; align-items: flex-start; gap: 16px; margin-bottom: 20px;">
                    <div style="width: 48px; height: 48px; border-radius: 14px; background: rgba(245, 158, 11, 0.14); color: #d97706; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                        <i data-lucide="camera-off" style="width: 24px; height: 24px;"></i>
                    </div>
                    <div style="flex: 1;">
                        <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 4px;">
                            <h3 style="font-size: 19px; font-weight: 700; color: var(--ink); margin: 0;">No Food Detected</h3>
                            <span class="badge" style="background: rgba(245, 158, 11, 0.16); color: #d97706; font-weight: 700; font-size: 11px; padding: 2px 10px; border-radius: var(--radius-pill);">Non-Food Image</span>
                        </div>
                        <p style="font-size: 14px; color: var(--muted); margin: 0; line-height: 1.5;">
                            Our computer vision pipeline inspected your photo, but could not detect any recognizable food, plated meals, or edible ingredients.
                        </p>
                    </div>
                </div>

                <div style="background: rgba(245, 158, 11, 0.08); border-left: 3px solid #f59e0b; padding: 12px 16px; border-radius: 6px; margin-bottom: 20px; font-size: 13px; color: var(--ink);">
                    <strong style="color: #d97706;">Detection Reason:</strong> ${reason}
                    ${hint ? `<div style="margin-top: 6px; font-size: 12px; color: var(--muted);">Hint entered: <em>"${escapeHTML(hint)}"</em> (could not match visible food in photo)</div>` : ''}
                </div>

                <div style="background: var(--pale); border: 1px solid var(--line); border-radius: var(--radius-md); padding: 18px; margin-bottom: 24px;">
                    <h4 style="font-size: 13px; font-weight: 700; color: var(--ink); text-transform: uppercase; letter-spacing: 0.5px; margin: 0 0 10px 0; display: flex; align-items: center; gap: 8px;">
                        <i data-lucide="lightbulb" style="width: 16px; height: 16px; color: #d97706;"></i>
                        Tips for Accurate Food Recognition
                    </h4>
                    <ul style="margin: 0; padding-left: 20px; font-size: 13px; color: var(--muted); line-height: 1.7;">
                        <li><strong>Overhead View:</strong> Take the photo directly from above looking down at the plate (30–50 cm away).</li>
                        <li><strong>Keep Plate Rim Visible:</strong> Keep the full plate or bowl edges inside the frame so the physical scale can be calibrated.</li>
                        <li><strong>Adequate Lighting:</strong> Avoid deep shadows, flash reflections, or extreme angles.</li>
                        <li><strong>Clean Background:</strong> Keep clutter, phones, cutlery wrappers, or packaging off the plate area.</li>
                    </ul>
                </div>

                <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                    <button type="button" class="btn btn-forest" onclick="window.retryScannerUpload()" style="display: inline-flex; align-items: center; gap: 8px;">
                        <i data-lucide="upload" style="width: 16px; height: 16px;"></i>
                        <span>Upload Another Photo</span>
                    </button>
                    <button type="button" class="btn btn-outline" onclick="window.openModal('modal-add-meal')" style="display: inline-flex; align-items: center; gap: 8px; background: var(--paper-card); color: var(--ink); border-color: var(--line);">
                        <i data-lucide="plus-circle" style="width: 16px; height: 16px;"></i>
                        <span>Add Meal Manually</span>
                    </button>
                    <button type="button" class="btn btn-outline" onclick="window.loadDemoAnalysis()" style="display: inline-flex; align-items: center; gap: 8px; background: var(--paper-card); color: var(--ink); border-color: var(--line);">
                        <i data-lucide="sparkles" style="width: 16px; height: 16px; color: var(--lime);"></i>
                        <span>Try Sample Plate</span>
                    </button>
                </div>
            </div>
        `;
        renderLucide();
        clearPlateSpatialOverlay();
        resultDiv.scrollIntoView({ behavior: "smooth" });
    }

    window.renderNonFoodDetected = renderNonFoodDetected;

    // -------------------------------------------------------------
    // OVERHEAD DISPLAY BOX OVERLAY CONTROLS (CLEAN DISPLAY)
    // -------------------------------------------------------------
    function clearPlateSpatialOverlay() {
        const wrapper = document.getElementById("plateOverlayWrapper");
        const toolbar = document.getElementById("overlayToolbar");
        if (wrapper) wrapper.style.display = "none";
        if (toolbar) toolbar.style.display = "none";
    }

    function renderPlateSpatialOverlay() {
        // Segmentation overlay intentionally disabled on overhead preview for a clean, unobstructed photo view
        clearPlateSpatialOverlay();
    }

    function highlightCluster() {}

    function focusIngredientCard(ingredientName) {
        if (!ingredientName) return;
        const enc = encodeURIComponent(ingredientName);
        const clean = escapeHTML(ingredientName);
        let inputEl = document.getElementById(`weight_input_${enc}`) || document.getElementById(`weight_input_${clean}`) || document.getElementById(`weight_input_${ingredientName}`);
        if (!inputEl && window.currentIngredients) {
            const lower = ingredientName.toLowerCase();
            const matchedKey = Object.keys(window.currentIngredients).find(k =>
                k.toLowerCase().includes(lower) || lower.includes(k.toLowerCase())
            );
            if (matchedKey) {
                const matchEnc = encodeURIComponent(matchedKey);
                inputEl = document.getElementById(`weight_input_${matchEnc}`) || document.getElementById(`weight_input_${escapeHTML(matchedKey)}`) || document.getElementById(`weight_input_${matchedKey}`);
            }
        }
        if (inputEl) {
            inputEl.scrollIntoView({ behavior: "smooth", block: "center" });
            inputEl.focus();
            inputEl.select();
            showToast(`📍 Focused on ${clean}`);
        }
    }

    window.toggleFoodPolygons = function () {};
    window.toggleFoodPins = function () {};
    window.clearPlateSpatialOverlay = clearPlateSpatialOverlay;
    window.renderPlateSpatialOverlay = renderPlateSpatialOverlay;

    // -------------------------------------------------------------
    // RENDER ANALYSIS RESULTS: DATA STRIP, MACRO SPLIT, TRAY & TABLES
    // -------------------------------------------------------------
    function renderAnalysisResults(data) {
        const resultDiv = document.getElementById("result");
        if (!resultDiv) return;

        if (!data || data.is_food === false) {
            renderNonFoodDetected(data || {});
            return;
        }

        const dishName = escapeHTML(data.dish || window.currentDish || "Plated Meal");
        const scale = data.scale_reference || {};
        const plateText = scale.detected
            ? `Calibrated — ${scale.plate_diameter_cm || 25} cm plate`
            : `${scale.plate_diameter_cm || 25} cm plate (partially visible / estimated)`;

        const totals = data.recipe_totals || {
            total_weight_g: 0, energy_kcal: 0, carb_g: 0, protein_g: 0, fat_g: 0, fibre_g: 0
        };

        // Caloric macro proportions (Carb: 4 kcal/g, Prot: 4 kcal/g, Fat: 9 kcal/g)
        const carbKcal = Math.max(0, totals.carb_g * 4);
        const protKcal = Math.max(0, totals.protein_g * 4);
        const fatKcal  = Math.max(0, totals.fat_g * 9);
        const sumMacroKcal = (carbKcal + protKcal + fatKcal) || 1;

        const carbPct = Math.round((carbKcal / sumMacroKcal) * 100);
        const protPct = Math.round((protKcal / sumMacroKcal) * 100);
        const fatPct  = Math.max(0, 100 - carbPct - protPct);

        // 1. Data strip
        const totalsHtml = `
            <div class="result-data-strip">
                <div class="data-cell data-cell-primary">
                    <div class="data-cell-label">Calories</div>
                    <div class="data-cell-value">${totals.energy_kcal}<span class="data-cell-unit">kcal</span></div>
                </div>
                <div class="data-cell">
                    <div class="data-cell-label">Weight</div>
                    <div class="data-cell-value">${totals.total_weight_g}<span class="data-cell-unit">g</span></div>
                </div>
                <div class="data-cell">
                    <div class="data-cell-label">Carbs</div>
                    <div class="data-cell-value">${totals.carb_g}<span class="data-cell-unit">g</span></div>
                </div>
                <div class="data-cell">
                    <div class="data-cell-label">Protein</div>
                    <div class="data-cell-value">${totals.protein_g}<span class="data-cell-unit">g</span></div>
                </div>
                <div class="data-cell">
                    <div class="data-cell-label">Fats</div>
                    <div class="data-cell-value">${totals.fat_g}<span class="data-cell-unit">g</span></div>
                </div>
                <div class="data-cell">
                    <div class="data-cell-label">Fibre</div>
                    <div class="data-cell-value">${totals.fibre_g}<span class="data-cell-unit">g</span></div>
                </div>
            </div>
        `;

        // 2. Caloric macro distribution bar
        const macroSplitHtml = `
            <div class="macro-dist-box">
                <div class="macro-dist-header">
                    <span>Caloric split</span>
                    <span>${totals.energy_kcal} kcal total</span>
                </div>
                <div class="macro-split-bar">
                    <div class="bar-seg-carbs" style="width: ${carbPct}%" title="Carbohydrates: ${carbPct}%"></div>
                    <div class="bar-seg-prot" style="width: ${protPct}%" title="Protein: ${protPct}%"></div>
                    <div class="bar-seg-fat" style="width: ${fatPct}%" title="Fats: ${fatPct}%"></div>
                </div>
                <div class="macro-legend">
                    <div class="legend-item">
                        <span class="legend-dot" style="background: var(--carb-color)"></span>
                        <span>Carbs: ${totals.carb_g}g (${carbPct}%)</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-dot" style="background: var(--protein-color)"></span>
                        <span>Protein: ${totals.protein_g}g (${protPct}%)</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-dot" style="background: var(--fat-color)"></span>
                        <span>Fats: ${totals.fat_g}g (${fatPct}%)</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-dot" style="background: var(--fibre-color)"></span>
                        <span>Fibre: ${totals.fibre_g}g</span>
                    </div>
                </div>
            </div>
        `;

        // 3. Interactive ingredient tray
        let ingrRowsHtml = "";
        const reviewRequired = Boolean(data.review_required);
        for (const [name, weight] of Object.entries(window.currentIngredients)) {
            const cleanName = escapeHTML(name);
            const encName = encodeURIComponent(name);
            const emoji = getFoodEmoji(name);
            ingrRowsHtml += `
                <div class="ingr-chip-row">
                    <div class="ingr-name-col">
                        <span class="food-emoji">${emoji}</span>
                        ${reviewRequired
                            ? `<input type="text" class="input-add-name" value="${cleanName}" aria-label="Food name" onchange="window.renameIngredient(decodeURIComponent('${encName}'), this.value)">`
                            : `<span>${cleanName}</span>`}
                    </div>
                    <div class="ingr-stepper-col">
                        <button type="button" class="btn-step" onclick="window.adjustWeight(decodeURIComponent('${encName}'), -10)" title="Subtract 10g">−</button>
                        <input type="number" id="weight_input_${encName}" class="input-gram" value="${weight}" min="5" step="5" onchange="window.updateWeightManual(decodeURIComponent('${encName}'), this.value)">
                        <span class="unit-label">g</span>
                        <button type="button" class="btn-step" onclick="window.adjustWeight(decodeURIComponent('${encName}'), 10)" title="Add 10g">+</button>
                        <button type="button" class="btn-remove-ingr" onclick="window.removeIngredient(decodeURIComponent('${encName}'))" title="Remove item">✕</button>
                    </div>
                </div>
            `;
        }

        const reviewNotice = reviewRequired ? `
            <div style="margin: 0 0 14px; padding: 12px 14px; border: 1px solid #f59e0b; border-radius: 10px; background: rgba(245, 158, 11, .09); color: var(--ink); font-size: 13px; line-height: 1.45;">
                <strong>Review detected foods before logging.</strong> Rename, remove, or add foods below, then recalculate nutrition.
                ${(data.review_reasons || []).length ? `<div style="margin-top: 5px; color: var(--muted);">${escapeHTML(data.review_reasons.join(" "))}</div>` : ''}
            </div>` : '';
        const interactiveTrayHtml = `
            <div class="interactive-tray-container">
                <div class="tray-title-row">
                    <h4>Meal components</h4>
                    <span class="tray-subtext">${reviewRequired ? 'Confirm foods and portions before recalculating' : 'Adjust portion weights before recalculating'}</span>
                </div>
                ${reviewNotice}
                <div class="ingredient-chips-list">
                    ${ingrRowsHtml || '<p style="color:var(--muted);font-size:13px;padding:8px 0">No food items listed.</p>'}
                </div>
                <div class="tray-add-row">
                    <input type="text" id="newIngrName" class="input-add-name" placeholder="Add food item (e.g. Roti, Dal, Salad)…">
                    <input type="number" id="newIngrWeight" class="input-add-weight" placeholder="100" value="100" min="5" step="5">
                    <span class="unit-label">g</span>
                    <button type="button" class="btn-add-ingr" onclick="window.addCustomIngredient()">
                        <i data-lucide="plus" class="icon-sm"></i> Add
                    </button>
                </div>
                <button type="button" id="recalcBtn" class="btn-recalc-nutrition" onclick="window.recalculateNutrition()">
                    <i data-lucide="calculator" class="icon-sm"></i>
                    <span>Recalculate nutrition</span>
                </button>
            </div>
        `;

        // 4. Nutrient breakdown table
        let regressionRows = "";
        const nutritionList = data.nutrition || [];
        nutritionList.forEach((item) => {
            regressionRows += `
                <tr>
                    <td><strong>${escapeHTML(item.ingredient)}</strong></td>
                    <td>${item.weight_g} g</td>
                    <td><strong>${item.energy_kcal} kcal</strong></td>
                    <td>${item.carb_g} g</td>
                    <td>${item.protein_g} g</td>
                    <td>${item.fat_g} g</td>
                    <td>${item.fibre_g} g</td>
                </tr>
            `;
        });

        const regressionTableHtml = `
            <p class="result-section-title">Nutrient breakdown</p>
            <div class="table-responsive">
                <table class="results-table">
                    <thead>
                        <tr>
                            <th>Food item</th>
                            <th>Portion</th>
                            <th>Calories</th>
                            <th>Carbs</th>
                            <th>Protein</th>
                            <th>Fats</th>
                            <th>Fibre</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${regressionRows || '<tr><td colspan="7">No nutritional values calculated.</td></tr>'}
                    </tbody>
                </table>
            </div>
        `;

        // 5. Action toolbar
        const actionToolbarHtml = `
            <div class="report-action-bar">
                <button type="button" class="btn-report btn-report-primary" onclick="window.saveScannedMealToDiary()">
                    <i data-lucide="bookmark-check" class="icon-sm"></i>
                    <span>Save to Diary</span>
                </button>
                <button type="button" class="btn-report" onclick="window.copyMealSummary()">
                    <i data-lucide="copy" class="icon-sm"></i>
                    <span>Copy summary</span>
                </button>
                <button type="button" class="btn-report" onclick="window.printMealReport()">
                    <i data-lucide="printer" class="icon-sm"></i>
                    <span>Print report</span>
                </button>
                <button type="button" class="btn-report" onclick="window.clearUploads()">
                    <i data-lucide="rotate-ccw" class="icon-sm"></i>
                    <span>Analyze another</span>
                </button>
            </div>
        `;

        // 6. Medical Disclaimer Card
        const disclaimerHtml = `
            <div class="disclaimer-card">
                <span class="disclaimer-badge">Medical Disclaimer</span>
                <p>Nutritional calculations and weight estimates are automated approximations based on visual modeling and regression. They are not intended or recommended for medical, diagnostic, or clinical dietary purposes.</p>
            </div>
        `;

        const vlmProvider = data.vlm_provider || (data.image_processing && data.image_processing.vlm_provider);
        const vlmKeyUsed = data.vlm_key_used || (data.image_processing && data.image_processing.vlm_key_used);
        const providerBadge = vlmProvider ? `
            <span class="badge" style="background: rgba(16, 185, 129, 0.12); color: #059669; border: 1px solid rgba(16, 185, 129, 0.25); font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 6px; display: inline-flex; align-items: center; gap: 4px;">
                <i data-lucide="sparkles" style="width: 12px; height: 12px;"></i>
                <span>${escapeHTML(vlmProvider)}${vlmKeyUsed && vlmKeyUsed !== 'N/A' ? ` (${escapeHTML(vlmKeyUsed)})` : ''}</span>
            </span>
        ` : '';

        // Combine into DOM
        resultDiv.innerHTML = `
            <div class="result-card-header" style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 8px;">
                <div>
                    <h3 style="margin: 0 0 4px;">${dishName}</h3>
                    <div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
                        <span class="scale-badge">${plateText}</span>
                        ${providerBadge}
                    </div>
                </div>
            </div>
            ${totalsHtml}
            ${macroSplitHtml}
            ${interactiveTrayHtml}
            ${regressionTableHtml}
            ${actionToolbarHtml}
            ${disclaimerHtml}
        `;

        renderLucide();
        renderPlateSpatialOverlay(data.spatial_clusters || []);
    }

    function getImageThumbnailDataUrl(imgEl, maxDim = 256) {
        if (!imgEl) return null;
        try {
            const naturalW = imgEl.naturalWidth || imgEl.videoWidth || imgEl.width || 0;
            const naturalH = imgEl.naturalHeight || imgEl.videoHeight || imgEl.height || 0;
            if (naturalW <= 0 || naturalH <= 0) return null;
            const canvas = document.createElement('canvas');
            let w = naturalW;
            let h = naturalH;
            if (w > h) {
                if (w > maxDim) {
                    h = Math.round((h * maxDim) / w);
                    w = maxDim;
                }
            } else {
                if (h > maxDim) {
                    w = Math.round((w * maxDim) / h);
                    h = maxDim;
                }
            }
            canvas.width = Math.max(1, w);
            canvas.height = Math.max(1, h);
            const ctx = canvas.getContext('2d');
            ctx.drawImage(imgEl, 0, 0, canvas.width, canvas.height);
            return canvas.toDataURL('image/jpeg', 0.72);
        } catch (e) {
            console.warn("Unable to convert preview to persistent thumbnail:", e);
            return null;
        }
    }

    // -------------------------------------------------------------
    // ACTION HANDLERS: SAVE TO DIARY, COPY SUMMARY, PRINT REPORT
    // -------------------------------------------------------------
    function saveScannedMealToDiary() {
        if (!window.currentAnalysisData) {
            showToast("No analyzed meal available to save.", true);
            return;
        }

        const totals = window.currentAnalysisData.recipe_totals || {};
        const now = new Date();
        const timeStr = now.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });

        const hr = now.getHours();
        let mealType = 'Lunch';
        if (hr < 11) mealType = 'Breakfast';
        else if (hr >= 16 && hr < 21) mealType = 'Dinner';
        else if (hr >= 21 || (hr >= 14 && hr < 16)) mealType = 'Snack';

        const topPreview = document.getElementById('topPreview');
        let imgSrc = "assets/nourish-bowl.png";
        if (topPreview && topPreview.style.display !== 'none' && topPreview.src && !topPreview.src.startsWith('blob:null')) {
            const thumb = getImageThumbnailDataUrl(topPreview, 256);
            if (thumb) {
                imgSrc = thumb;
            } else if (!topPreview.src.startsWith('blob:')) {
                imgSrc = topPreview.src;
            }
        }

        const newMeal = {
            id: `meal-${Date.now()}`,
            name: window.currentDish || "Plated Meal",
            type: mealType,
            time: timeStr,
            date: getTodayDateString(0),
            calories: Math.round(totals.energy_kcal || 450),
            protein: Math.round(totals.protein_g || 20),
            carbs: Math.round(totals.carb_g || 45),
            fat: Math.round(totals.fat_g || 15),
            fiber: Math.round(totals.fibre_g || 6),
            image: imgSrc,
            components: (window.currentAnalysisData.nutrition || []).map(item => ({
                name: item.ingredient,
                mass: item.weight_g,
                calories: item.energy_kcal,
                p: item.protein_g,
                c: item.carb_g,
                f: item.fat_g
            }))
        };

        Storage.addMeal(newMeal);
        renderDashboard();
        renderScanner();
        showToast(`✦ "${newMeal.name}" (${newMeal.calories} kcal) saved to today's diary!`);

        const saveBanner = document.createElement('div');
        saveBanner.className = 'card';
        saveBanner.style.background = 'var(--pale)';
        saveBanner.style.borderColor = 'var(--line-strong)';
        saveBanner.style.marginTop = '18px';
        saveBanner.style.display = 'flex';
        saveBanner.style.alignItems = 'center';
        saveBanner.style.justifyContent = 'space-between';
        saveBanner.style.flexWrap = 'wrap';
        saveBanner.style.gap = '12px';
        saveBanner.innerHTML = `
            <div style="display: flex; align-items: center; gap: 10px;">
                <i data-lucide="check-circle" style="color: var(--forest); width: 22px; height: 22px;"></i>
                <div>
                    <b style="font-size: 14px; color: var(--ink);">Saved to Today's Diary!</b>
                    <div class="tiny" style="color: var(--muted);">${newMeal.name} · ${newMeal.calories} kcal added to daily totals.</div>
                </div>
            </div>
            <button type="button" class="btn btn-sm btn-forest" onclick="window.nutriScanSwitchView('dashboard')">
                <i data-lucide="layout-dashboard" class="icon-sm"></i>
                <span>View Dashboard</span>
            </button>
        `;
        const resDiv = document.getElementById('result');
        if (resDiv) {
            resDiv.insertBefore(saveBanner, resDiv.firstChild);
            renderLucide();
        }
    }

    window.saveScannedMealToDiary = saveScannedMealToDiary;

    window.printMealReport = function () {
        window.print();
    };

    window.copyMealSummary = function () {
        if (!window.currentAnalysisData) return;
        const d = window.currentAnalysisData;
        const t = d.recipe_totals || {};
        let text = `NutriScanAI — Meal summary\n`;
        text += `Meal: ${d.dish || window.currentDish || "Plated Meal"}\n`;
        text += `Total Weight: ${t.total_weight_g || 0}g | Calories: ${t.energy_kcal || 0} kcal\n`;
        text += `Carbs: ${t.carb_g || 0}g | Protein: ${t.protein_g || 0}g | Fats: ${t.fat_g || 0}g | Fibre: ${t.fibre_g || 0}g\n\n`;
        text += `Meal Components:\n`;
        for (const [name, weight] of Object.entries(window.currentIngredients || {})) {
            text += `- ${name}: ${weight}g\n`;
        }
        navigator.clipboard.writeText(text).then(() => {
            showToast("Meal summary copied to clipboard!");
        }).catch(err => {
            console.error("Clipboard copy failed:", err);
            showToast("Summary copied!");
        });
    };

    window.nutriScanSelectRecentMeal = function (id) {
        const meal = Storage.getMeals().find(m => m.id === id);
        if (!meal) return;

        window.currentDish = meal.name;
        window.currentIngredients = {};
        if (Array.isArray(meal.components) && meal.components.length > 0) {
            meal.components.forEach(c => {
                window.currentIngredients[c.name] = c.mass || 80;
            });
        } else {
            window.currentIngredients[meal.name] = 250;
        }

        const topPreview = document.getElementById('topPreview');
        const topPlaceholder = document.getElementById('topPlaceholder');
        if (topPreview) {
            topPreview.src = meal.image || 'assets/nourish-bowl.png';
            topPreview.style.display = 'block';
        }
        if (topPlaceholder) topPlaceholder.style.display = 'none';
        const cont = topPreview?.closest('.upload-preview');
        if (cont) cont.classList.add('has-image');

        const hint = document.getElementById('scanFoodHint');
        if (hint) hint.value = meal.name;

        window.currentAnalysisData = {
            dish: meal.name,
            scale_reference: { detected: true, plate_diameter_cm: 25.0 },
            ingredients: { ...window.currentIngredients },
            nutrition: (meal.components || []).map(c => ({
                ingredient: c.name,
                weight_g: c.mass || 80,
                energy_kcal: c.calories || 120,
                carb_g: c.c || 15,
                protein_g: c.p || 8,
                fat_g: c.f || 4,
                fibre_g: 2
            })),
            recipe_totals: {
                total_weight_g: (meal.components || []).reduce((acc, c) => acc + (c.mass || 80), 0) || 300,
                energy_kcal: meal.calories,
                carb_g: meal.carbs,
                protein_g: meal.protein,
                fat_g: meal.fat,
                fibre_g: meal.fiber || 6
            }
        };

        renderAnalysisResults(window.currentAnalysisData);
        showToast(`Loaded "${meal.name}" into scanner`);
        document.getElementById('result')?.scrollIntoView({ behavior: 'smooth' });
    };

    // -------------------------------------------------------------
    // SCREEN 3: MEAL HISTORY RENDERER & FILTERS
    // -------------------------------------------------------------
    let currentHistoryFilter = 'today';
    let currentTypeFilter = 'all';
    let currentHistorySearch = '';

    function renderHistory() {
        const meals = Storage.getMeals();
        const container = document.getElementById('history-meals-container');
        const countBadge = document.getElementById('history-count-badge');
        const groupHeader = document.getElementById('history-group-header');
        if (!container) return;

        const todayStr = getTodayDateString(0);
        const d7Str = getTodayDateString(7);
        const d30Str = getTodayDateString(30);

        // Date range filtering
        let filtered = meals;
        if (currentHistoryFilter === 'today') {
            filtered = meals.filter(m => m.date === todayStr);
            if (groupHeader) groupHeader.textContent = "Today's Log";
        } else if (currentHistoryFilter === 'week') {
            filtered = meals.filter(m => m.date >= d7Str);
            if (groupHeader) groupHeader.textContent = "This Week's Log";
        } else if (currentHistoryFilter === 'month') {
            filtered = meals.filter(m => m.date >= d30Str);
            if (groupHeader) groupHeader.textContent = "This Month's Log";
        } else {
            if (groupHeader) groupHeader.textContent = "All Logged Meals";
        }

        // Meal type secondary filter
        if (currentTypeFilter !== 'all') {
            filtered = filtered.filter(m => m.type === currentTypeFilter);
        }

        // Search text filter
        if (currentHistorySearch) {
            const query = currentHistorySearch.toLowerCase();
            filtered = filtered.filter(m =>
                (m.name && m.name.toLowerCase().includes(query)) ||
                (m.type && m.type.toLowerCase().includes(query))
            );
        }

        if (countBadge) {
            countBadge.textContent = `${filtered.length} meal${filtered.length === 1 ? '' : 's'} logged`;
        }

        if (filtered.length === 0) {
            container.innerHTML = `
                <div style="padding: 30px; text-align: center; color: var(--muted);">
                    <i data-lucide="inbox" class="icon-lg" style="margin-bottom: 8px;"></i>
                    <p>No meals match your active filters.</p>
                </div>
            `;
        } else {
            container.innerHTML = filtered.map(m => `
                <div class="meal" data-meal-id="${m.id}" title="Click to view meal details">
                    <img class="thumb" src="${escapeHTML(m.image || 'assets/nourish-bowl.png')}" alt="${escapeHTML(m.name)}" onerror="this.src='assets/nourish-bowl.png'">
                    <div class="meal-info">
                        <div class="meal-name">${escapeHTML(m.name)}</div>
                        <div class="meal-time">
                            <i data-lucide="calendar" class="icon-sm"></i>
                            <span>${escapeHTML(m.date)} · ${escapeHTML(m.type)} · ${escapeHTML(m.time || '12:00 PM')}</span>
                        </div>
                        <div class="meal-macros-inline">
                            <span>Protein: ${m.protein || 0}g</span>
                            <span>•</span>
                            <span>Carbs: ${m.carbs || 0}g</span>
                            <span>•</span>
                            <span>Fat: ${m.fat || 0}g</span>
                        </div>
                    </div>
                    <div class="meal-kcal">
                        ${m.calories || 0}<small>kcal</small>
                    </div>
                    <div class="meal-actions" onclick="event.stopPropagation();">
                        <button class="del-meal-btn" data-delete-id="${m.id}" title="Delete meal">
                            <i data-lucide="trash-2" class="icon-sm"></i>
                        </button>
                    </div>
                </div>
            `).join('');
        }

        // Summary cards
        const weekMeals = meals.filter(m => m.date >= d7Str);
        document.getElementById('hist-stat-week-count').textContent = `${weekMeals.length} meals`;

        const counts = {};
        meals.forEach(m => {
            counts[m.name] = (counts[m.name] || 0) + 1;
        });
        let favName = "Garden nourish bowl";
        let maxCount = 0;
        for (const name in counts) {
            if (counts[name] > maxCount) {
                maxCount = counts[name];
                favName = name;
            }
        }
        document.getElementById('hist-stat-fav-name').textContent = favName;

        const dayKcalMap = {};
        weekMeals.forEach(m => {
            dayKcalMap[m.date] = (dayKcalMap[m.date] || 0) + (Number(m.calories) || 0);
        });
        const activeDays = Object.keys(dayKcalMap).length || 1;
        const totalWeekKcal = Object.values(dayKcalMap).reduce((a, b) => a + b, 0);
        const avgKcal = Math.round(totalWeekKcal / activeDays);
        document.getElementById('hist-stat-avg-kcal').textContent = `${avgKcal.toLocaleString()} kcal`;
    }

    function initHistoryEvents() {
        // Date filters
        document.querySelectorAll('[data-history-filter]').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('[data-history-filter]').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentHistoryFilter = btn.getAttribute('data-history-filter');
                renderHistory();
                renderLucide();
            });
        });

        // Meal type secondary filters
        document.querySelectorAll('[data-type-filter]').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('[data-type-filter]').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentTypeFilter = btn.getAttribute('data-type-filter');
                renderHistory();
                renderLucide();
            });
        });

        // Search input
        const searchInput = document.getElementById('history-search-input');
        if (searchInput) {
            searchInput.addEventListener('input', e => {
                currentHistorySearch = e.target.value.trim();
                renderHistory();
                renderLucide();
            });
        }

        // Global meal row click delegation -> Opens Interactive Meal Details
        document.addEventListener('click', e => {
            const mealRow = e.target.closest('.meal[data-meal-id]');
            if (mealRow && !e.target.closest('.del-meal-btn')) {
                const id = mealRow.getAttribute('data-meal-id');
                openMealDetails(id);
            }
        });

        // Delete meal button delegation
        document.addEventListener('click', e => {
            const delBtn = e.target.closest('[data-delete-id]');
            if (delBtn) {
                const id = delBtn.getAttribute('data-delete-id');
                Storage.deleteMeal(id);
                showToast('Meal removed from diary');
                renderDashboard();
                renderHistory();
                renderReports();
                renderLucide();
            }
        });
    }

    function openMealDetails(mealId) {
        const meal = Storage.getMeals().find(m => m.id === mealId);
        if (!meal) return;

        const goals = Storage.getGoals();

        document.getElementById('detail-meal-title').textContent = `${meal.type} Details`;
        document.getElementById('detail-meal-img').src = meal.image || "assets/nourish-bowl.png";
        document.getElementById('detail-meal-name').textContent = meal.name;
        document.getElementById('detail-meal-time').textContent = `${meal.type} · ${meal.time || '12:00 PM'} · ${meal.date}`;
        document.getElementById('detail-meal-kcal').innerHTML = `${meal.calories}<small>kcal</small>`;

        const pPct = Math.round((meal.protein / (goals.protein || 100)) * 100);
        const cPct = Math.round((meal.carbs / (goals.carbs || 220)) * 100);
        const fPct = Math.round((meal.fat / (goals.fat || 65)) * 100);

        document.getElementById('detail-meal-p').textContent = `${meal.protein || 0}g`;
        document.getElementById('detail-meal-p-pct').textContent = `${pPct}% of target`;

        document.getElementById('detail-meal-c').textContent = `${meal.carbs || 0}g`;
        document.getElementById('detail-meal-c-pct').textContent = `${cPct}% of target`;

        document.getElementById('detail-meal-f').textContent = `${meal.fat || 0}g`;
        document.getElementById('detail-meal-f-pct').textContent = `${fPct}% of target`;

        document.getElementById('detail-meal-fib').textContent = `${meal.fiber || 6}g`;

        const compBody = document.getElementById('detail-components-body');
        if (compBody) {
            if (Array.isArray(meal.components) && meal.components.length > 0) {
                compBody.innerHTML = meal.components.map(c => `
                    <tr>
                        <td><b>${escapeHTML(c.name)}</b></td>
                        <td>${c.mass || 80}g</td>
                        <td>${c.calories || 100} kcal</td>
                        <td>${c.p || 0}g / ${c.c || 0}g / ${c.f || 0}g</td>
                    </tr>
                `).join('');
            } else {
                compBody.innerHTML = `
                    <tr>
                        <td><b>${escapeHTML(meal.name)}</b></td>
                        <td>~300g</td>
                        <td>${meal.calories} kcal</td>
                        <td>${meal.protein}g / ${meal.carbs}g / ${meal.fat}g</td>
                    </tr>
                `;
            }
        }

        const btnDel = document.getElementById('btn-detail-delete');
        if (btnDel) {
            btnDel.onclick = () => {
                Storage.deleteMeal(meal.id);
                closeModal('modal-meal-details');
                showToast(`"${meal.name}" deleted`);
                renderDashboard();
                renderHistory();
                renderReports();
            };
        }

        openModal('modal-meal-details');
    }

    // -------------------------------------------------------------
    // SCREEN 4: DIET REPORTS RENDERER
    // -------------------------------------------------------------
    let currentReportDays = 7;

    function renderReports() {
        const meals = Storage.getMeals();
        const goals = Storage.getGoals();
        const cutoffDate = getTodayDateString(currentReportDays);

        const filtered = meals.filter(m => m.date >= cutoffDate);

        const dayMap = {};
        let totalKcal = 0;
        let totalProtein = 0;
        let totalCarbs = 0;
        let totalFat = 0;

        filtered.forEach(m => {
            dayMap[m.date] = (dayMap[m.date] || 0) + (Number(m.calories) || 0);
            totalKcal += Number(m.calories) || 0;
            totalProtein += Number(m.protein) || 0;
            totalCarbs += Number(m.carbs) || 0;
            totalFat += Number(m.fat) || 0;
        });

        const activeDaysCount = Object.keys(dayMap).length || 1;
        const avgKcal = Math.round(totalKcal / activeDaysCount) || 1860;
        const avgProtein = Math.round(totalProtein / activeDaysCount) || 78;

        document.getElementById('rep-avg-kcal').innerHTML = `${avgKcal.toLocaleString()} <span style="font-size: 13px; font-weight: 600; color: var(--muted);">kcal / day</span>`;
        document.getElementById('rep-avg-protein').innerHTML = `${avgProtein}g <span style="font-size: 13px; font-weight: 600; color: var(--muted);">daily avg</span>`;
        document.getElementById('rep-logged-days').innerHTML = `${activeDaysCount} <span style="font-size: 13px; font-weight: 600; color: var(--muted);">active days</span>`;

        renderReportChart(meals, goals.calories || 2000, currentReportDays);

        const macroSum = (totalProtein * 4) + (totalCarbs * 4) + (totalFat * 9) || 1000;
        const pPct = Math.round(((totalProtein * 4) / macroSum) * 100) || 32;
        const cPct = Math.round(((totalCarbs * 4) / macroSum) * 100) || 40;
        const fPct = Math.max(0, 100 - pPct - cPct) || 28;

        document.getElementById('rep-donut-protein').textContent = `${pPct}% (${Math.round(totalProtein)}g)`;
        document.getElementById('rep-donut-carbs').textContent = `${cPct}% (${Math.round(totalCarbs)}g)`;
        document.getElementById('rep-donut-fat').textContent = `${fPct}% (${Math.round(totalFat)}g)`;

        const donutEl = document.getElementById('reports-macro-donut');
        if (donutEl) {
            donutEl.style.background = `conic-gradient(#8CBF35 0 ${pPct}%, #EBCB73 ${pPct}% ${pPct + cPct}%, #A9C7D0 ${pPct + cPct}% 100%)`;
        }
    }

    function renderReportChart(meals, targetKcal, numDays) {
        const chartEl = document.getElementById('reports-calorie-chart');
        if (!chartEl) return;

        const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

        const barsData = [];
        for (let i = numDays - 1; i >= 0; i--) {
            const d = new Date();
            d.setDate(d.getDate() - i);
            const dateStr = d.toISOString().split('T')[0];
            const dayName = dayNames[d.getDay()];

            const dayTotal = meals
                .filter(m => m.date === dateStr)
                .reduce((acc, m) => acc + (Number(m.calories) || 0), 0);

            barsData.push({
                name: dayName,
                date: dateStr,
                kcal: dayTotal,
                isToday: i === 0
            });
        }

        const maxVal = Math.max(targetKcal * 1.2, ...barsData.map(b => b.kcal), 2200);

        chartEl.innerHTML = barsData.map(b => {
            const hPct = Math.max(8, Math.min(100, Math.round((b.kcal / maxVal) * 100)));
            return `
                <div class="barcol">
                    <div class="bar ${b.isToday ? 'hi' : ''}" style="height: ${hPct}%;">
                        <span class="bar-val">${b.kcal} kcal</span>
                    </div>
                    <span class="day">${b.name}</span>
                </div>
            `;
        }).join('');
    }

    function initReportEvents() {
        document.querySelectorAll('[data-report-days]').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('[data-report-days]').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentReportDays = parseInt(btn.getAttribute('data-report-days'), 10) || 7;
                renderReports();
                renderLucide();
            });
        });

        const btnExport = document.getElementById('btn-export-report');
        if (btnExport) {
            btnExport.addEventListener('click', () => {
                const report = {
                    app: "NutriScanAI",
                    user: Storage.getProfile(),
                    goals: Storage.getGoals(),
                    exportedAt: new Date().toISOString(),
                    meals: Storage.getMeals()
                };
                downloadJSON(report, `nutriscanai_report_${getTodayDateString(0)}.json`);
                showToast('NutriScanAI summary downloaded!');
            });
        }
    }

    // -------------------------------------------------------------
    // SCREEN 5: GOALS & PREFERENCES RENDERER
    // -------------------------------------------------------------
    function renderGoals() {
        const goals = Storage.getGoals();
        const meals = Storage.getMeals();
        const todayStr = getTodayDateString(0);

        let todayKcal = 0;
        let todayProtein = 0;
        let todayCarbs = 0;
        let todayFat = 0;

        meals.filter(m => m.date === todayStr).forEach(m => {
            todayKcal += Number(m.calories) || 0;
            todayProtein += Number(m.protein) || 0;
            todayCarbs += Number(m.carbs) || 0;
            todayFat += Number(m.fat) || 0;
        });

        // Set Values & Progress Bars
        document.getElementById('goal-val-calories').textContent = `${goals.calories.toLocaleString()} kcal`;
        document.getElementById('goal-val-protein').textContent = `${goals.protein} g`;
        document.getElementById('goal-val-carbs').textContent = `${goals.carbs} g`;
        document.getElementById('goal-val-fat').textContent = `${goals.fat} g`;
        document.getElementById('goal-val-water').textContent = `${goals.water} L`;

        const kPct = Math.min(100, Math.round((todayKcal / goals.calories) * 100));
        const pPct = Math.min(100, Math.round((todayProtein / goals.protein) * 100));
        const cPct = Math.min(100, Math.round((todayCarbs / goals.carbs) * 100));
        const fPct = Math.min(100, Math.round((todayFat / goals.fat) * 100));

        const waterCurrent = Storage.getWater().ml / 1000;
        const wPct = Math.min(100, Math.round((waterCurrent / (goals.water || 2.5)) * 100));

        document.getElementById('goal-prog-calories').style.width = `${kPct}%`;
        document.getElementById('goal-prog-protein').style.width = `${pPct}%`;
        document.getElementById('goal-prog-carbs').style.width = `${cPct}%`;
        document.getElementById('goal-prog-fat').style.width = `${fPct}%`;
        document.getElementById('goal-prog-water').style.width = `${wPct}%`;

        // Update dietary chips
        const activePatterns = goals.dietaryPatterns || [];
        document.querySelectorAll('#goals-dietary-chips .diet-tag').forEach(tag => {
            const tagName = tag.getAttribute('data-tag');
            if (activePatterns.includes(tagName)) {
                tag.classList.add('active');
            } else {
                tag.classList.remove('active');
            }
        });

        const allergensPrev = document.getElementById('goals-allergens-preview');
        if (allergensPrev) {
            allergensPrev.textContent = goals.allergens || "None specified";
        }

        const schedulePrev = document.getElementById('goals-schedule-preview');
        if (schedulePrev) {
            schedulePrev.textContent = goals.schedule || "Breakfast: 9:00 AM · Lunch: 1:00 PM · Dinner: 7:30 PM";
        }
    }

    function initGoalsEvents() {
        // Toggle dietary tag chips
        document.querySelectorAll('#goals-dietary-chips .diet-tag').forEach(tag => {
            tag.addEventListener('click', () => {
                tag.classList.toggle('active');
                const active = [];
                document.querySelectorAll('#goals-dietary-chips .diet-tag.active').forEach(t => {
                    active.push(t.getAttribute('data-tag'));
                });
                const goals = Storage.getGoals();
                goals.dietaryPatterns = active;
                Storage.saveGoals(goals);
                showToast('Food preferences updated!');
                renderProfile();
            });
        });

        // Edit Targets Modal Trigger
        const btnOpenTargetModal = document.getElementById('btn-open-target-modal');
        if (btnOpenTargetModal) {
            btnOpenTargetModal.addEventListener('click', () => {
                const goals = Storage.getGoals();
                document.getElementById('edit-goal-calories').value = goals.calories;
                document.getElementById('edit-goal-protein').value = goals.protein;
                document.getElementById('edit-goal-carbs').value = goals.carbs;
                document.getElementById('edit-goal-fat').value = goals.fat;
                document.getElementById('edit-goal-water').value = goals.water;
                openModal('modal-targets');
            });
        }

        // Auto calculate TDEE from Goals
        const btnCalcTdee = document.getElementById('btn-calc-tdee-goals');
        const btnWizardRecalc = document.getElementById('btn-wizard-recalc');

        function triggerBiometricCalculation() {
            const profile = Storage.getProfile() || {
                weight: 65,
                height: 170,
                age: 25,
                gender: 'Female',
                activity: 'light',
                goal: 'Maintain weight'
            };
            const calc = calculateTDEE(
                profile.weight || 65,
                profile.height || 170,
                profile.age || 25,
                profile.gender || 'Female',
                profile.activity || 'light',
                profile.goal || 'Maintain weight'
            );

            document.getElementById('edit-goal-calories').value = calc.calories;
            document.getElementById('edit-goal-protein').value = calc.protein;
            document.getElementById('edit-goal-carbs').value = calc.carbs;
            document.getElementById('edit-goal-fat').value = calc.fat;
            document.getElementById('edit-goal-water').value = calc.water;

            showToast(`Calculated: ${calc.calories} kcal based on ${profile.gender || 'biometrics'}, ${profile.weight || 65}kg, ${profile.height || 170}cm`);
        }

        if (btnCalcTdee) {
            btnCalcTdee.addEventListener('click', () => {
                openModal('modal-targets');
                triggerBiometricCalculation();
            });
        }
        if (btnWizardRecalc) {
            btnWizardRecalc.addEventListener('click', triggerBiometricCalculation);
        }

        // Save Targets Button
        const btnSaveTargets = document.getElementById('btn-save-targets');
        if (btnSaveTargets) {
            btnSaveTargets.addEventListener('click', () => {
                const goals = Storage.getGoals();
                goals.calories = parseInt(document.getElementById('edit-goal-calories').value, 10) || 2000;
                goals.protein = parseInt(document.getElementById('edit-goal-protein').value, 10) || 100;
                goals.carbs = parseInt(document.getElementById('edit-goal-carbs').value, 10) || 220;
                goals.fat = parseInt(document.getElementById('edit-goal-fat').value, 10) || 65;
                goals.water = parseFloat(document.getElementById('edit-goal-water').value) || 2.5;

                Storage.saveGoals(goals);
                closeModal('modal-targets');
                showToast('Daily targets saved!');
                renderDashboard();
                renderGoals();
            });
        }

        // Wire Allergens Edit Row
        const rowAllergens = document.getElementById('row-allergens-edit');
        if (rowAllergens) {
            rowAllergens.addEventListener('click', () => {
                const goals = Storage.getGoals();
                document.getElementById('edit-allergens-input').value = goals.allergens || '';
                openModal('modal-allergens');
            });
        }

        const btnSaveAllergens = document.getElementById('btn-save-allergens');
        if (btnSaveAllergens) {
            btnSaveAllergens.addEventListener('click', () => {
                const goals = Storage.getGoals();
                goals.allergens = (document.getElementById('edit-allergens-input').value || '').trim();
                Storage.saveGoals(goals);
                closeModal('modal-allergens');
                showToast('Allergens and foods to avoid updated!');
                renderGoals();
            });
        }

        // Wire Meal Schedule Edit Row
        const rowSchedule = document.getElementById('row-schedule-edit');
        if (rowSchedule) {
            rowSchedule.addEventListener('click', () => {
                openModal('modal-schedule');
            });
        }

        const btnSaveSchedule = document.getElementById('btn-save-schedule');
        if (btnSaveSchedule) {
            btnSaveSchedule.addEventListener('click', () => {
                const b = document.getElementById('sched-breakfast').value.trim() || '9:00 AM';
                const l = document.getElementById('sched-lunch').value.trim() || '1:00 PM';
                const d = document.getElementById('sched-dinner').value.trim() || '7:30 PM';

                const goals = Storage.getGoals();
                goals.schedule = `Breakfast: ${b} · Lunch: ${l} · Dinner: ${d}`;
                Storage.saveGoals(goals);
                closeModal('modal-schedule');
                showToast('Preferred meal schedule saved!');
                renderGoals();
            });
        }
    }

    // -------------------------------------------------------------
    // SCREEN 6: PROFILE & BIOMETRICS RENDERER & ACTIONS
    // -------------------------------------------------------------
    function renderProfile() {
        const profile = Storage.getProfile() || {
            name: "NutriScan User",
            avatar: "N",
            tagline: "Local storage mode",
            gender: "Female",
            age: 25,
            weight: 65,
            height: 170,
            goal: "Maintain weight & vitality"
        };
        const goals = Storage.getGoals();
        const meals = Storage.getMeals();

        const avatarInitial = (profile.name || "N").trim().charAt(0).toUpperCase();

        // Update Topbar and Sidebar Profile Chips
        const topbarAvatar = document.getElementById('topbar-avatar');
        const topbarName = document.getElementById('topbar-name');
        const sideAvatar = document.getElementById('side-avatar');
        const sideName = document.getElementById('side-name');

        if (topbarAvatar) topbarAvatar.textContent = avatarInitial;
        if (topbarName) topbarName.textContent = profile.name.split(' ')[0];
        if (sideAvatar) sideAvatar.textContent = avatarInitial;
        if (sideName) sideName.textContent = profile.name;

        // Profile View Elements
        const heroAvatar = document.getElementById('profile-hero-avatar');
        const heroName = document.getElementById('profile-hero-name');
        const heroBio = document.getElementById('profile-hero-bio');
        const heroGoal = document.getElementById('profile-hero-goal');
        const bioText = document.getElementById('profile-biometrics-text');
        const dietaryText = document.getElementById('profile-dietary-text');
        const recCount = document.getElementById('profile-records-count');

        if (heroAvatar) heroAvatar.textContent = avatarInitial;
        if (heroName) heroName.textContent = profile.name;
        if (heroBio) heroBio.textContent = `${profile.tagline || 'Local offline mode'} · ${profile.gender || 'Female'} · ${profile.age || 25} yrs`;
        if (heroGoal) heroGoal.textContent = profile.goal || "Maintain weight & vitality";

        if (bioText) bioText.textContent = `Weight: ${profile.weight || 65} kg · Height: ${profile.height || 170} cm · Age: ${profile.age || 25} · ${profile.gender || 'Female'}`;
        if (dietaryText) dietaryText.textContent = (goals.dietaryPatterns || []).join(', ') || 'No restrictions';
        if (recCount) recCount.textContent = `${meals.length} meals saved in browser localStorage`;
    }

    function initProfileEvents() {
        // Open Edit Profile Modal
        function openProfileEditor() {
            const profile = Storage.getProfile() || {
                name: "",
                age: 25,
                gender: "Female",
                weight: 65,
                height: 170,
                activity: "light",
                goal: "Maintain weight & vitality"
            };
            document.getElementById('edit-prof-name').value = profile.name || "";
            document.getElementById('edit-prof-age').value = profile.age || 25;
            document.getElementById('edit-prof-gender').value = profile.gender || "Female";
            document.getElementById('edit-prof-weight').value = profile.weight || 65;
            document.getElementById('edit-prof-height').value = profile.height || 170;
            document.getElementById('edit-prof-activity').value = profile.activity || "light";
            document.getElementById('edit-prof-goal').value = profile.goal || "Maintain weight & vitality";
            openModal('modal-profile');
        }

        const btnOpenProfile = document.getElementById('btn-open-profile-modal');
        if (btnOpenProfile) btnOpenProfile.addEventListener('click', openProfileEditor);

        const rowBiometrics = document.getElementById('profile-biometrics-row');
        if (rowBiometrics) rowBiometrics.addEventListener('click', openProfileEditor);

        const rowDietary = document.getElementById('profile-dietary-row');
        if (rowDietary) {
            rowDietary.addEventListener('click', () => switchView('goals'));
        }

        // Save Profile & Recalculate TDEE
        const btnSaveProfile = document.getElementById('btn-save-profile');
        if (btnSaveProfile) {
            btnSaveProfile.addEventListener('click', () => {
                const profile = Storage.getProfile() || {};
                profile.name = document.getElementById('edit-prof-name').value.trim() || "NutriScan User";
                profile.age = parseInt(document.getElementById('edit-prof-age').value, 10) || 25;
                profile.gender = document.getElementById('edit-prof-gender').value;
                profile.weight = parseFloat(document.getElementById('edit-prof-weight').value) || 65;
                profile.height = parseInt(document.getElementById('edit-prof-height').value, 10) || 170;
                profile.activity = document.getElementById('edit-prof-activity').value;
                profile.goal = document.getElementById('edit-prof-goal').value;

                Storage.saveProfile(profile);

                // Auto sync daily targets to updated biometrics
                const calc = calculateTDEE(
                    profile.weight,
                    profile.height,
                    profile.age,
                    profile.gender,
                    profile.activity,
                    profile.goal
                );

                const goals = Storage.getGoals();
                goals.calories = calc.calories;
                goals.protein = calc.protein;
                goals.carbs = calc.carbs;
                goals.fat = calc.fat;
                goals.water = calc.water;
                Storage.saveGoals(goals);

                closeModal('modal-profile');
                showToast(`Profile saved! Daily targets synced to ${calc.calories} kcal`);
                renderProfile();
                renderDashboard();
                renderGoals();
            });
        }

        // Retest Backend Ping
        const btnRetest = document.getElementById('btn-retest-backend');
        if (btnRetest) {
            btnRetest.addEventListener('click', async () => {
                showToast('Pinging NutriScanAI backend services...');
                await checkBackendStatus();
                if (backendOnline) {
                    showToast('FastAPI backend online & responding on port 8000!');
                } else {
                    showToast('Backend offline. Operating in standalone localStorage mode.', true);
                }
            });
        }

        // Export All Data JSON
        const btnExportJson = document.getElementById('btn-export-json');
        if (btnExportJson) {
            btnExportJson.addEventListener('click', () => {
                const data = {
                    app: "NutriScanAI",
                    version: "2.0",
                    exportedAt: new Date().toISOString(),
                    profile: Storage.getProfile(),
                    goals: Storage.getGoals(),
                    water: Storage.getWater(),
                    meals: Storage.getMeals()
                };
                downloadJSON(data, `nutriscanai_backup_${getTodayDateString(0)}.json`);
                showToast('Complete diary data backup exported!');
            });
        }

        // Reset Sample Data
        const btnResetData = document.getElementById('btn-reset-data');
        if (btnResetData) {
            btnResetData.addEventListener('click', () => {
                if (confirm('Reset your meals, targets, and profile to NutriScanAI default sample data?')) {
                    Storage.resetAll();
                    showToast('All data reset to default seed!');
                    switchView('dashboard');
                }
            });
        }
    }

    // -------------------------------------------------------------
    // MODAL DIALOGS HELPER
    // -------------------------------------------------------------
    function openModal(modalId, uncloseable = false) {
        const modal = document.getElementById(modalId);
        if (modal) {
            if (uncloseable) {
                modal.classList.add('uncloseable');
            } else {
                modal.classList.remove('uncloseable');
            }
            modal.classList.add('open');
            renderLucide();
        }
    }

    function closeModal(modalId, force = false) {
        const modal = document.getElementById(modalId);
        if (modal) {
            if (!force && modal.classList.contains('uncloseable')) {
                return; // Protected: modal cannot be closed while uncloseable
            }
            modal.classList.remove('open');
            if (modalId === 'modal-camera') {
                stopCameraStream();
            }
        }
    }

    window.openModal = openModal;
    window.closeModal = closeModal;

    function initModals() {
        document.addEventListener('click', e => {
            const closeTarget = e.target.closest('[data-close-modal]');
            if (closeTarget) {
                const modalId = closeTarget.getAttribute('data-close-modal');
                const modal = document.getElementById(modalId);
                if (modal && modal.classList.contains('uncloseable')) {
                    return; // Cannot dismiss uncloseable modal
                }
                closeModal(modalId);
            }

            // Click outside modal to close
            if (e.target.classList.contains('modal-overlay')) {
                if (e.target.classList.contains('uncloseable')) {
                    return; // Cannot dismiss uncloseable modal
                }
                closeModal(e.target.id);
            }
        });

        // Close on Escape key if modal is not uncloseable
        document.addEventListener('keydown', e => {
            if (e.key === 'Escape') {
                const openModals = document.querySelectorAll('.modal-overlay.open');
                openModals.forEach(m => {
                    if (!m.classList.contains('uncloseable')) {
                        closeModal(m.id);
                    }
                });
            }
        });

        // Add Manual Meal Trigger & Save
        const btnAddManual = document.getElementById('btn-add-manual-meal');
        if (btnAddManual) {
            btnAddManual.addEventListener('click', () => {
                document.getElementById('manual-meal-name').value = '';
                document.getElementById('manual-meal-calories').value = '';
                document.getElementById('manual-meal-protein').value = '';
                document.getElementById('manual-meal-carbs').value = '';
                document.getElementById('manual-meal-fat').value = '';
                openModal('modal-add-meal');
            });
        }

        const btnSaveManual = document.getElementById('btn-save-manual-meal');
        if (btnSaveManual) {
            btnSaveManual.addEventListener('click', () => {
                const name = (document.getElementById('manual-meal-name').value || '').trim();
                const type = document.getElementById('manual-meal-type').value;
                const kcal = parseInt(document.getElementById('manual-meal-calories').value, 10);
                const p = parseInt(document.getElementById('manual-meal-protein').value, 10) || 0;
                const c = parseInt(document.getElementById('manual-meal-carbs').value, 10) || 0;
                const f = parseInt(document.getElementById('manual-meal-fat').value, 10) || 0;

                if (!name || isNaN(kcal)) {
                    showToast('Please enter a meal name and calories', true);
                    return;
                }

                const now = new Date();
                const timeStr = now.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });

                const newMeal = {
                    id: `manual-${Date.now()}`,
                    name,
                    type,
                    time: timeStr,
                    date: getTodayDateString(0),
                    calories: kcal,
                    protein: p,
                    carbs: c,
                    fat: f,
                    fiber: 4,
                    image: "assets/nourish-bowl.png",
                    components: [{ name, mass: 200, calories: kcal, p, c, f }]
                };

                Storage.addMeal(newMeal);
                closeModal('modal-add-meal');
                showToast(`Logged "${name}" to diary!`);
                renderDashboard();
                renderHistory();
            });
        }
    }

    function downloadJSON(obj, filename) {
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(obj, null, 2));
        const dlAnchor = document.createElement('a');
        dlAnchor.setAttribute("href", dataStr);
        dlAnchor.setAttribute("download", filename);
        document.body.appendChild(dlAnchor);
        dlAnchor.click();
        dlAnchor.remove();
    }

    // -------------------------------------------------------------
    // NAVIGATION LINK WIRING
    // -------------------------------------------------------------
    function initNavigation() {
        // Desktop sidebar navitems
        document.querySelectorAll('.sidebar .navitem').forEach(el => {
            el.addEventListener('click', () => {
                const v = el.getAttribute('data-view');
                if (v) switchView(v);
            });
        });

        // Mobile bottom nav buttons
        document.querySelectorAll('.mobile-bottom-nav .mob-nav-btn, .mobile-bottom-nav .mob-nav-scan').forEach(el => {
            el.addEventListener('click', () => {
                const v = el.getAttribute('data-view');
                if (v) switchView(v);
            });
        });

        // In-page shortcuts
        document.getElementById('dash-cta-scan-btn')?.addEventListener('click', () => switchView('scanner'));
        document.getElementById('dash-view-history-link')?.addEventListener('click', () => switchView('history'));
        document.getElementById('dash-view-reports-link')?.addEventListener('click', () => switchView('reports'));
        document.getElementById('scan-view-all-history')?.addEventListener('click', () => switchView('history'));
        document.getElementById('side-profile-trigger')?.addEventListener('click', () => switchView('profile'));
        document.getElementById('topbar-userchip')?.addEventListener('click', () => window.nutriScanOpenAuthModal());
        document.getElementById('btn-quick-search')?.addEventListener('click', () => {
            switchView('history');
            document.getElementById('history-search-input')?.focus();
        });
    }

    // -------------------------------------------------------------
    // DOB & LIVE AGE CALCULATION
    // -------------------------------------------------------------
    function calculateAgeFromDOB(dobString) {
        if (!dobString) return null;
        const birthDate = new Date(dobString);
        if (isNaN(birthDate.getTime())) return null;
        const today = new Date();
        let age = today.getFullYear() - birthDate.getFullYear();
        const m = today.getMonth() - birthDate.getMonth();
        if (m < 0 || (m === 0 && today.getDate() < birthDate.getDate())) {
            age--;
        }
        return age;
    }

    window.handleDOBChange = function (inputId, badgeId) {
        const input = document.getElementById(inputId);
        const badge = document.getElementById(badgeId);
        if (!input || !badge) return;
        const val = input.value;
        if (!val) {
            badge.textContent = 'Select DOB';
            badge.classList.remove('calculated');
            return;
        }
        const age = calculateAgeFromDOB(val);
        if (age === null || age < 1 || age > 120) {
            badge.textContent = 'Invalid date';
            badge.classList.remove('calculated');
        } else {
            badge.textContent = `Age: ${age} yrs`;
            badge.classList.add('calculated');
        }
    };

    // -------------------------------------------------------------
    // PASSWORD EYE TOGGLE
    // -------------------------------------------------------------
    window.togglePasswordEye = function (inputId, btn) {
        const input = document.getElementById(inputId);
        if (!input) return;
        const isPassword = input.type === 'password';
        input.type = isPassword ? 'text' : 'password';
        const iconName = isPassword ? 'eye-off' : 'eye';
        btn.innerHTML = `<i data-lucide="${iconName}"></i>`;
        renderLucide();
    };

    // -------------------------------------------------------------
    // PASSWORD MATCH VALIDATOR
    // -------------------------------------------------------------
    window.handleCheckPasswordMatch = function () {
        const pass = document.getElementById('auth-signup-password')?.value || '';
        const confirmPass = document.getElementById('auth-signup-confirm-password')?.value || '';
        const msg = document.getElementById('signup-pass-match-msg');
        if (!msg) return;

        if (!confirmPass) {
            msg.style.display = 'none';
            return;
        }
        msg.style.display = 'block';
        if (pass === confirmPass) {
            msg.style.color = '#10B981';
            msg.innerHTML = '✓ Passwords match';
        } else {
            msg.style.color = '#EF4444';
            msg.innerHTML = '✕ Passwords do not match';
        }
    };

    // -------------------------------------------------------------
    // SUPABASE AUTH & CLOUD MODAL CONTROLLERS
    // -------------------------------------------------------------
    window.nutriScanOpenAuthModal = function (initialTab = 'signup', uncloseable = false) {
        openModal('modal-auth', uncloseable);
        window.switchAuthTab(initialTab);
    };

    window.switchAuthTab = function (tabName) {
        const tabs = ['signup', 'signin'];
        tabs.forEach(t => {
            const btn = document.getElementById(`tab-auth-${t}`);
            const pane = document.getElementById(`auth-pane-${t}`);
            if (btn) btn.classList.toggle('active', t === tabName);
            if (pane) pane.style.display = (t === tabName) ? 'block' : 'none';
        });
        renderLucide();
    };

    window.openLocalSignupModal = function () {
        const isUncloseable = document.getElementById('modal-auth')?.classList.contains('uncloseable');
        document.getElementById('modal-auth')?.classList.remove('open');
        openModal('modal-local-signup', isUncloseable);
    };

    window.backToAuthModal = function () {
        const isUncloseable = document.getElementById('modal-local-signup')?.classList.contains('uncloseable');
        document.getElementById('modal-local-signup')?.classList.remove('open');
        openModal('modal-auth', isUncloseable);
        window.switchAuthTab('signup');
    };

    window.nutriScanOpenConfigModal = function () {
        openModal('modal-supabase-config');
        if (window.NutriScanSupabaseConfig) {
            const cfg = window.NutriScanSupabaseConfig.get();
            const urlInput = document.getElementById('cfg-supabase-url');
            const keyInput = document.getElementById('cfg-supabase-key');
            if (urlInput) urlInput.value = cfg.supabaseUrl;
            if (keyInput) keyInput.value = cfg.publishableKey;
        }
    };

    window.handleLocalSignupSubmit = function () {
        const name = (document.getElementById('local-signup-name')?.value || '').trim();
        const dob = document.getElementById('local-signup-dob')?.value || '';
        const height = parseInt(document.getElementById('local-signup-height')?.value, 10);
        const weight = parseFloat(document.getElementById('local-signup-weight')?.value);
        const gender = document.getElementById('local-signup-gender')?.value || 'Female';
        const activity = document.getElementById('local-signup-activity')?.value || 'light';
        const goal = document.getElementById('local-signup-goal')?.value || 'Maintain weight & vitality';

        if (!name) {
            showToast('Please enter your full name.', true);
            return;
        }
        if (!dob) {
            showToast('Please enter your date of birth.', true);
            return;
        }
        const age = calculateAgeFromDOB(dob);
        if (age === null || age < 1 || age > 120) {
            showToast('Please enter a valid date of birth.', true);
            return;
        }
        if (isNaN(height) || height < 50 || height > 260) {
            showToast('Please enter a valid height in cm.', true);
            return;
        }
        if (isNaN(weight) || weight < 20 || weight > 300) {
            showToast('Please enter a valid weight in kg.', true);
            return;
        }

        // Calculate scientific BMR/TDEE
        const calc = calculateTDEE(weight, height, age, gender, activity, goal);

        // Build and save local profile
        const localProfile = {
            name: name,
            avatar: name.charAt(0).toUpperCase(),
            tagline: "Local offline mode",
            dob: dob,
            age: age,
            gender: gender,
            weight: weight,
            height: height,
            activity: activity,
            goal: goal,
            mode: "local",
            createdAt: new Date().toISOString()
        };

        Storage.saveProfile(localProfile);
        Storage.saveGoals({
            calories: calc.calories,
            protein: calc.protein,
            carbs: calc.carbs,
            fat: calc.fat,
            water: calc.water,
            streak: 1,
            dietaryPatterns: [],
            allergens: "None",
            schedule: "Breakfast: 8:30 AM · Lunch: 1:00 PM · Dinner: 7:30 PM"
        });

        localStorage.setItem('nutriscan_user_mode', 'local');
        localStorage.setItem('nutriscan_local_onboarded', 'true');

        // Unlock modals
        const authM = document.getElementById('modal-auth');
        const localM = document.getElementById('modal-local-signup');
        if (authM) authM.classList.remove('uncloseable', 'open');
        if (localM) localM.classList.remove('uncloseable', 'open');

        window.checkLoginState();
        showToast(`✦ Welcome to NutriScanAI, ${name}! (Local Mode)`);
        renderDashboard();
        renderProfile();
    };

    window.handleAuthSignIn = async function () {
        const email = (document.getElementById('auth-signin-email')?.value || '').trim();
        const password = document.getElementById('auth-signin-password')?.value || '';
        const btn = document.getElementById('btn-submit-signin');

        if (!email || !password) {
            showToast('Please enter both email and password.', true);
            return;
        }
        if (!window.NutriScanSync || !window.NutriScanSync.auth) {
            showToast("Supabase client is not loaded.", true);
            return;
        }

        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<i data-lucide="loader-2" class="icon-sm"></i> <span>Signing in…</span>';
            renderLucide();
        }

        try {
            await window.NutriScanSync.auth.signIn(email, password);
            localStorage.setItem('nutriscan_user_mode', 'cloud');

            const authM = document.getElementById('modal-auth');
            if (authM) authM.classList.remove('uncloseable', 'open');

            showToast(`✦ Welcome back! Connected to NutriScanAI.`);
            window.checkLoginState();
            renderDashboard();
            renderProfile();
        } catch (err) {
            console.error("Sign in error:", err);
            showToast(err.message || "Failed to sign in. Check email & password.", true);
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = '<span>Log In</span>';
            }
        }
    };

    window.handleAuthSignUp = async function () {
        const name = (document.getElementById('auth-signup-name')?.value || '').trim();
        const dob = document.getElementById('auth-signup-dob')?.value || '';
        const height = parseInt(document.getElementById('auth-signup-height')?.value, 10);
        const weight = parseFloat(document.getElementById('auth-signup-weight')?.value);
        const email = (document.getElementById('auth-signup-email')?.value || '').trim();
        const password = document.getElementById('auth-signup-password')?.value || '';
        const confirmPassword = document.getElementById('auth-signup-confirm-password')?.value || '';
        const btn = document.getElementById('btn-submit-signup');

        if (!name) {
            showToast('Please enter your full name.', true);
            return;
        }
        if (!dob) {
            showToast('Please provide your date of birth.', true);
            return;
        }
        const age = calculateAgeFromDOB(dob);
        if (age === null || age < 1 || age > 120) {
            showToast('Please enter a valid date of birth.', true);
            return;
        }
        if (isNaN(height) || height < 50 || height > 260) {
            showToast('Please enter a valid height in cm.', true);
            return;
        }
        if (isNaN(weight) || weight < 20 || weight > 300) {
            showToast('Please enter a valid weight in kg.', true);
            return;
        }
        if (!email || !email.includes('@')) {
            showToast('Please enter a valid email address.', true);
            return;
        }
        if (password.length < 6) {
            showToast('Password must be at least 6 characters.', true);
            return;
        }
        if (password !== confirmPassword) {
            showToast('Passwords do not match. Please verify.', true);
            return;
        }
        if (!window.NutriScanSync || !window.NutriScanSync.auth) {
            showToast("Supabase client is not loaded.", true);
            return;
        }

        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<i data-lucide="loader-2" class="icon-sm"></i> <span>Creating account…</span>';
            renderLucide();
        }

        try {
            const data = await window.NutriScanSync.auth.signUp(email, password, {
                full_name: name,
                dob: dob,
                age: age,
                height: height,
                weight: weight
            });

            // Calculate scientific TDEE and store locally
            const calc = calculateTDEE(weight, height, age, 'Female', 'light', 'Maintain weight & vitality');
            const userProfile = {
                name: name,
                avatar: name.charAt(0).toUpperCase(),
                tagline: "Cloud Synced",
                dob: dob,
                age: age,
                gender: "Female",
                weight: weight,
                height: height,
                activity: "light",
                goal: "Maintain weight & vitality",
                mode: "cloud",
                createdAt: new Date().toISOString()
            };

            Storage.saveProfile(userProfile);
            Storage.saveGoals({
                calories: calc.calories,
                protein: calc.protein,
                carbs: calc.carbs,
                fat: calc.fat,
                water: calc.water,
                streak: 1,
                dietaryPatterns: [],
                allergens: "None",
                schedule: "Breakfast: 8:30 AM · Lunch: 1:00 PM · Dinner: 7:30 PM"
            });

            localStorage.setItem('nutriscan_user_mode', 'cloud');

            const authM = document.getElementById('modal-auth');
            if (authM) authM.classList.remove('uncloseable', 'open');

            if (data.session) {
                showToast(`✦ Welcome to NutriScanAI, ${name}! Account created.`);
            } else {
                showToast(`✦ Account created! Please check your email to confirm, or log in.`, false);
            }

            window.checkLoginState();
            renderDashboard();
            renderProfile();
        } catch (err) {
            console.error("Sign up error:", err);
            showToast(err.message || "Failed to create account.", true);
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = '<span>Create Free Account</span>';
            }
        }
    };

    window.handleForgotPassword = async function () {
        const email = prompt("Enter your registered email address to receive a password reset link:");
        if (!email) return;

        try {
            await window.NutriScanSync.auth.resetPassword(email);
            showToast(`Password recovery link sent to ${email}`);
        } catch (err) {
            showToast(err.message || "Could not send reset link.", true);
        }
    };

    window.handleSaveSupabaseConfig = function () {
        const url = document.getElementById('cfg-supabase-url')?.value;
        const key = document.getElementById('cfg-supabase-key')?.value;
        if (!url || !key) {
            showToast("Please provide both Supabase URL and Key", true);
            return;
        }

        if (window.NutriScanSupabaseConfig) {
            window.NutriScanSupabaseConfig.save(url, key);
            if (window.NutriScanSync) {
                window.NutriScanSync.init();
            }
            showToast("✦ Supabase credentials saved! Reconnecting...");
            closeModal('modal-supabase-config');
        }
    };

    window.handleResetSupabaseConfig = function () {
        if (window.NutriScanSupabaseConfig) {
            window.NutriScanSupabaseConfig.reset();
            const cfg = window.NutriScanSupabaseConfig.defaults;
            const u = document.getElementById('cfg-supabase-url');
            const k = document.getElementById('cfg-supabase-key');
            if (u) u.value = cfg.supabaseUrl;
            if (k) k.value = cfg.publishableKey;
            if (window.NutriScanSync) window.NutriScanSync.init();
            showToast("Supabase config reset to default test project.");
        }
    };

    // -------------------------------------------------------------
    // GEMINI AI MULTI-KEY ROUND-ROBIN ROUTER CONTROLLERS
    // -------------------------------------------------------------
    async function checkGeminiStatus() {
        const descEl = document.getElementById('profile-gemini-desc');
        if (!descEl) return;

        try {
            const isPort8000 = (typeof window !== 'undefined' && window.location && window.location.port === "8000");
            const keysUrl = isPort8000 ? "/gemini/keys" : `${BACKEND_BASE_URL}/gemini/keys`;
            const resp = await fetch(keysUrl, { method: 'GET' });
            if (resp.ok) {
                const data = await resp.json();
                const total = data.total_keys || 0;
                const active = data.active_keys || 0;
                const model = data.model || 'gemini-2.5-flash';
                if (total === 0) {
                    descEl.textContent = "No keys configured — click Manage Keys to add";
                    descEl.style.color = "var(--muted)";
                } else {
                    descEl.textContent = `Active: ${active}/${total} key(s) (${model}) · Dynamic 429 failover`;
                    descEl.style.color = active > 0 ? "var(--forest, #059669)" : "var(--amber, #d97706)";
                }
            } else {
                descEl.textContent = "Vision service offline (port 8001)";
            }
        } catch (e) {
            descEl.textContent = "Key router standby (configure keys)";
        }
    }
    window.checkGeminiStatus = checkGeminiStatus;

    window.nutriScanOpenGeminiModal = function () {
        openModal('modal-gemini-config');
        window.loadGeminiKeysStatus();
    };

    window.loadGeminiKeysStatus = async function () {
        const listEl = document.getElementById('gemini-keys-list');
        const counterEl = document.getElementById('gemini-pool-counter');
        const modelSelect = document.getElementById('cfg-gemini-model');
        const feedbackEl = document.getElementById('gemini-test-feedback');
        if (feedbackEl) feedbackEl.style.display = 'none';

        if (listEl) {
            listEl.innerHTML = '<div class="tiny" style="color: var(--muted); text-align: center; padding: 10px 0;">Refreshing key pool...</div>';
        }

        try {
            const isPort8000 = (typeof window !== 'undefined' && window.location && window.location.port === "8000");
            const keysUrl = isPort8000 ? "/gemini/keys" : `${BACKEND_BASE_URL}/gemini/keys`;
            const resp = await fetch(keysUrl);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();

            const keys = data.keys || [];
            const total = data.total_keys || keys.length;
            const active = data.active_keys || 0;

            if (counterEl) {
                counterEl.textContent = `${active} Active / ${total} Total`;
                counterEl.className = active > 0 ? "badge badge-success" : "badge badge-warning";
            }

            if (modelSelect && data.model) {
                modelSelect.value = data.model;
            }

            const vaultInfo = data.vault || {};
            const vaultText = document.getElementById('gemini-vault-status-text');
            if (vaultText) {
                if (vaultInfo.table_synced) {
                    vaultText.textContent = "Supabase Vault Synced · AES-256 Fernet Encrypted";
                    vaultText.style.color = "#059669";
                } else {
                    vaultText.textContent = "AES-256 Encrypted · Supabase Vault Connected (Local Cache Active)";
                    vaultText.style.color = "var(--muted)";
                }
            }

            if (listEl) {
                if (keys.length === 0) {
                    listEl.innerHTML = `
                        <div class="tiny" style="color: var(--muted); text-align: center; padding: 12px 0;">
                            No Gemini keys currently registered in pool.<br>Paste your keys below to encrypt and store in Supabase.
                        </div>
                    `;
                } else {
                    listEl.innerHTML = keys.map((k) => {
                        const isCooldown = k.status === 'cooldown';
                        const isInvalid = k.status === 'invalid';
                        const statusColor = isInvalid ? '#ef4444' : (isCooldown ? '#f59e0b' : '#10b981');
                        const statusLabel = isInvalid ? 'Invalid' : (isCooldown ? `Cooldown (${k.cooldown_remaining_sec}s)` : 'Active');

                        return `
                            <div style="display: flex; align-items: center; justify-content: space-between; padding: 6px 10px; background: var(--paper-card, #fff); border: 1px solid var(--border-color, #e5e7eb); border-radius: 6px; font-size: 12px;">
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: ${statusColor};"></span>
                                    <code style="font-weight: 700; font-size: 12px;">${escapeHTML(k.masked_key)}</code>
                                    <span style="font-size: 10px; color: ${statusColor}; font-weight: 600;">[${statusLabel}]</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 10px;">
                                    <span class="tiny" style="color: var(--muted);" title="Success / Failures">
                                        ✓ ${k.success_count || 0} · ✗ ${k.failure_count || 0}
                                    </span>
                                    <button type="button" class="iconbtn" onclick="window.handleDeleteGeminiKey('${k.masked_key}')" title="Remove this key" style="color: #ef4444; width: 22px; height: 22px;">
                                        <i data-lucide="trash-2" style="width: 13px; height: 13px;"></i>
                                    </button>
                                </div>
                            </div>
                        `;
                    }).join('');
                    renderLucide();
                }
            }

            checkGeminiStatus();
        } catch (err) {
            if (listEl) {
                listEl.innerHTML = `
                    <div class="tiny" style="color: #ef4444; text-align: center; padding: 12px 0;">
                        Could not load Gemini keys (${escapeHTML(err.message)}). Ensure Vision Service (8001) is running.
                    </div>
                `;
            }
        }
    };

    window.handleSaveGeminiKeys = async function () {
        const inputEl = document.getElementById('cfg-gemini-new-key');
        if (!inputEl) return;
        const text = inputEl.value.trim();
        if (!text) {
            showToast("Please enter at least one Gemini API key", true);
            return;
        }

        const rawKeys = text.split(/[\n,;]+/).map(s => s.trim()).filter(Boolean);
        if (rawKeys.length === 0) {
            showToast("No valid keys found in input", true);
            return;
        }

        try {
            const isPort8000 = (typeof window !== 'undefined' && window.location && window.location.port === "8000");
            const keysUrl = isPort8000 ? "/gemini/keys" : `${BACKEND_BASE_URL}/gemini/keys`;
            const token = window.NutriScanSync && window.NutriScanSync.auth ? await window.NutriScanSync.auth.getToken() : null;
            const headers = { "Content-Type": "application/json" };
            if (token) headers["Authorization"] = `Bearer ${token}`;

            const resp = await fetch(keysUrl, {
                method: "POST",
                headers: headers,
                body: JSON.stringify({
                    keys: rawKeys,
                    action: "add",
                    user_id: window.NutriScanSync && window.NutriScanSync.auth ? window.NutriScanSync.auth.getUserId() : null
                })
            });

            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${resp.status}`);
            }

            inputEl.value = "";
            showToast(`✦ Encrypted & saved ${rawKeys.length} key(s) to Supabase Vault!`);
            window.loadGeminiKeysStatus();
        } catch (err) {
            showToast(`Failed to add keys: ${err.message}`, true);
        }
    };

    window.handleSyncSupabaseKeys = async function () {
        const syncBtn = document.getElementById('btn-gemini-sync-supabase');
        if (syncBtn) syncBtn.disabled = true;
        showToast("Syncing keys from Supabase Cloud Vault...");

        try {
            const isPort8000 = (typeof window !== 'undefined' && window.location && window.location.port === "8000");
            const syncUrl = isPort8000 ? "/gemini/sync" : `${BACKEND_BASE_URL}/gemini/sync`;
            const token = window.NutriScanSync && window.NutriScanSync.auth ? await window.NutriScanSync.auth.getToken() : null;
            const headers = {};
            if (token) headers["Authorization"] = `Bearer ${token}`;

            const resp = await fetch(syncUrl, { method: "POST", headers: headers });
            const data = await resp.json();
            if (resp.ok) {
                showToast(`✦ Synced ${data.synced_count || 0} key(s) from Supabase Vault!`);
            } else {
                showToast(`Supabase sync: ${data.detail || data.message || 'Complete'}`);
            }
            window.loadGeminiKeysStatus();
        } catch (err) {
            showToast(`Sync error: ${err.message}`, true);
        } finally {
            if (syncBtn) syncBtn.disabled = false;
        }
    };

    window.handleDeleteGeminiKey = async function (identifier) {
        if (!confirm(`Remove key ${identifier} from the router pool and Supabase vault?`)) return;

        try {
            const isPort8000 = (typeof window !== 'undefined' && window.location && window.location.port === "8000");
            const delUrl = isPort8000 ? `/gemini/keys/${encodeURIComponent(identifier)}` : `${BACKEND_BASE_URL}/gemini/keys/${encodeURIComponent(identifier)}`;
            const token = window.NutriScanSync && window.NutriScanSync.auth ? await window.NutriScanSync.auth.getToken() : null;
            const headers = {};
            if (token) headers["Authorization"] = `Bearer ${token}`;

            const resp = await fetch(delUrl, { method: "DELETE", headers: headers });

            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${resp.status}`);
            }

            showToast(`Removed key ${identifier}`);
            window.loadGeminiKeysStatus();
        } catch (err) {
            showToast(`Could not remove key: ${err.message}`, true);
        }
    };

    window.handleChangeGeminiModel = async function () {
        const modelSelect = document.getElementById('cfg-gemini-model');
        if (!modelSelect) return;
        const model = modelSelect.value;

        try {
            const isPort8000 = (typeof window !== 'undefined' && window.location && window.location.port === "8000");
            const url = isPort8000 ? "/gemini/model" : `${BACKEND_BASE_URL}/gemini/model`;
            const resp = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model: model })
            });
            if (resp.ok) {
                showToast(`Gemini model switched to ${model}`);
                checkGeminiStatus();
            }
        } catch (e) {
            console.warn("Could not switch Gemini model:", e);
        }
    };

    window.handleTestGeminiKeys = async function () {
        const feedbackEl = document.getElementById('gemini-test-feedback');
        const testBtn = document.getElementById('btn-gemini-test');

        if (feedbackEl) {
            feedbackEl.style.display = 'block';
            feedbackEl.style.color = 'var(--muted)';
            feedbackEl.innerHTML = '<span class="spin-icon">⏳</span> Pinging Google Gemini API via Round-Robin Router...';
        }
        if (testBtn) testBtn.disabled = true;

        const startTime = performance.now();
        try {
            const isPort8000 = (typeof window !== 'undefined' && window.location && window.location.port === "8000");
            const testUrl = isPort8000 ? "/gemini/test" : `${BACKEND_BASE_URL}/gemini/test`;
            const resp = await fetch(testUrl, { method: "POST" });
            const data = await resp.json();
            const elapsed = Math.round(performance.now() - startTime);

            if (resp.ok && data.status === "ok") {
                if (feedbackEl) {
                    feedbackEl.style.color = '#10b981';
                    feedbackEl.innerHTML = `✓ <b>Success:</b> Responded in <b>${elapsed}ms</b> using key <code>${escapeHTML(data.key_used || '')}</code> (${escapeHTML(data.model || '')})`;
                }
                showToast(`✦ Gemini connected in ${elapsed}ms!`);
            } else {
                if (feedbackEl) {
                    feedbackEl.style.color = '#ef4444';
                    feedbackEl.innerHTML = `✗ <b>Error:</b> ${escapeHTML(data.message || 'Test failed')}`;
                }
                showToast(`Gemini test failed: ${data.message || 'Error'}`, true);
            }
            window.loadGeminiKeysStatus();
        } catch (err) {
            if (feedbackEl) {
                feedbackEl.style.color = '#ef4444';
                feedbackEl.innerHTML = `✗ <b>Connection Error:</b> ${escapeHTML(err.message)}`;
            }
            showToast(`Gemini test failed: ${err.message}`, true);
        } finally {
            if (testBtn) testBtn.disabled = false;
        }
    };

    // -------------------------------------------------------------
    // CENTRALIZED LOGIN STATE CHECKER (Always checks Cloud & Local)
    // -------------------------------------------------------------
    window.checkLoginState = function () {
        const isCloud = Boolean(window.NutriScanSync && window.NutriScanSync.auth && window.NutriScanSync.auth.isAuthenticated());
        const userMode = localStorage.getItem('nutriscan_user_mode');
        const localProfile = Storage.getProfile();
        const hasLocal = (userMode === 'local' || localStorage.getItem('nutriscan_local_onboarded') === 'true') && localProfile && localProfile.name;

        const syncPill = document.getElementById('cloud-sync-pill');
        const syncDot = document.getElementById('cloud-sync-dot');
        const syncText = document.getElementById('cloud-sync-text');
        const topbarName = document.getElementById('topbar-name');
        const topbarAvatar = document.getElementById('topbar-avatar');
        const sideName = document.getElementById('side-name');
        const sideAvatar = document.getElementById('side-avatar');
        const sideTagline = document.getElementById('side-tagline');

        const authModal = document.getElementById('modal-auth');
        const localModal = document.getElementById('modal-local-signup');

        if (isCloud) {
            const user = window.NutriScanSync.auth.getCurrentUser();
            const displayName = user?.user_metadata?.full_name || user?.email?.split('@')[0] || (localProfile?.name) || 'User';
            const initial = displayName.charAt(0).toUpperCase();

            if (topbarName) topbarName.textContent = displayName.split(' ')[0];
            if (topbarAvatar) topbarAvatar.textContent = initial;
            if (sideName) sideName.textContent = displayName;
            if (sideAvatar) sideAvatar.textContent = initial;
            if (sideTagline) sideTagline.textContent = user?.email || 'Cloud Synced';

            if (syncDot) syncDot.className = 'status-dot dot-online';
            if (syncText) syncText.textContent = 'Cloud Synced';
            if (syncPill) {
                syncPill.title = `Connected to Supabase (${user?.email})`;
                syncPill.onclick = () => window.nutriScanOpenAuthModal('signin', false);
            }

            if (authModal) authModal.classList.remove('uncloseable');
            if (localModal) localModal.classList.remove('uncloseable');
        } else if (hasLocal) {
            const displayName = localProfile.name;
            const initial = displayName.charAt(0).toUpperCase();

            if (topbarName) topbarName.textContent = displayName.split(' ')[0];
            if (topbarAvatar) topbarAvatar.textContent = initial;
            if (sideName) sideName.textContent = displayName;
            if (sideAvatar) sideAvatar.textContent = initial;
            if (sideTagline) sideTagline.textContent = 'Local Offline Mode';

            if (syncDot) syncDot.className = 'status-dot dot-local';
            if (syncText) syncText.textContent = 'Local Mode';
            if (syncPill) {
                syncPill.title = 'Running locally on this device. Click to connect cloud account.';
                syncPill.onclick = () => window.nutriScanOpenAuthModal('signup', false);
            }

            if (authModal) authModal.classList.remove('uncloseable');
            if (localModal) localModal.classList.remove('uncloseable');
        } else {
            // Fresh / Unauthenticated / Un-onboarded device
            if (topbarName) topbarName.textContent = 'Account';
            if (topbarAvatar) topbarAvatar.textContent = 'N';
            if (sideName) sideName.textContent = 'NutriScan User';
            if (sideTagline) sideTagline.textContent = 'Setup required';

            if (syncDot) syncDot.className = 'status-dot dot-guest';
            if (syncText) syncText.textContent = 'Setup Required';
            if (syncPill) {
                syncPill.title = 'Get started with Cloud or Local Mode';
                syncPill.onclick = () => window.nutriScanOpenAuthModal('signup', true);
            }

            // Automatically show the login/signup modal and make it uncloseable!
            window.nutriScanOpenAuthModal('signup', true);
        }
        renderLucide();
    };

    window.nutriScanRefreshActiveView = function () {
        if (currentView === 'dashboard') renderDashboard();
        if (currentView === 'scanner') renderScanner();
        if (currentView === 'history') renderHistory();
        if (currentView === 'reports') renderReports();
        if (currentView === 'goals') renderGoals();
        if (currentView === 'profile') renderProfile();
        renderLucide();
    };

    function initPasswordMeter() {
        const passInput = document.getElementById('auth-signup-password');
        const fill = document.getElementById('password-strength-fill');
        if (!passInput || !fill) return;

        passInput.addEventListener('input', () => {
            const val = passInput.value;
            let score = 0;
            if (val.length >= 6) score += 30;
            if (val.length >= 9) score += 20;
            if (/[A-Z]/.test(val)) score += 20;
            if (/[0-9]/.test(val)) score += 15;
            if (/[^A-Za-z0-9]/.test(val)) score += 15;

            fill.style.width = `${Math.min(100, score)}%`;
            if (score < 50) fill.style.background = '#EF4444';
            else if (score < 75) fill.style.background = '#F59E0B';
            else fill.style.background = '#10B981';

            if (typeof window.handleCheckPasswordMatch === 'function') {
                window.handleCheckPasswordMatch();
            }
        });
    }

    // -------------------------------------------------------------
    // INITIALIZATION ON DOM READY
    // -------------------------------------------------------------
    document.addEventListener('DOMContentLoaded', () => {
        // Theme initialization
        initTheme();

        // Render initial Lucide icons
        renderLucide();

        // Initialize state & UI
        initNavigation();
        initScannerUploads();
        initHistoryEvents();
        initWaterEvents();
        initReportEvents();
        initGoalsEvents();
        initProfileEvents();
        initModals();
        initPasswordMeter();

        // Initial view load
        switchView('dashboard');

        // Always check login state (Local vs Cloud vs Unauthenticated)
        checkLoginState();

        // Check backend server status
        checkBackendStatus();
        checkGeminiStatus();

        // Periodic background ping to backend (every 30 seconds)
        setInterval(() => {
            checkBackendStatus();
            checkGeminiStatus();
        }, 30000);
    });

})();
