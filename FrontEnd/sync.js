/**
 * NutriScanAI — Production Supabase Authentication & Two-Tier Sync Engine
 * Manages Supabase Auth, PostgreSQL CRUD, Real-Time WebSockets,
 * Offline Mutation Queuing, and Client-Side Plate Image Compression.
 */

(function () {
    'use strict';

    let supabaseClient = null;
    let currentUser = null;
    let realtimeChannel = null;
    let isSyncing = false;

    const QUEUE_STORAGE_KEY = 'nutriscan_sync_queue';

    // ─────────────────────────────────────────────────────────────
    // 1. SUPABASE CLIENT INITIALIZATION
    // ─────────────────────────────────────────────────────────────
    function initClient() {
        if (!window.supabase || typeof window.supabase.createClient !== 'function') {
            console.warn("Supabase JS SDK not loaded. Operating in Standalone / Guest mode.");
            return null;
        }

        const config = window.NutriScanSupabaseConfig ? window.NutriScanSupabaseConfig.get() : null;
        if (!config || !config.supabaseUrl || !config.publishableKey) {
            console.warn("Supabase credentials not configured.");
            return null;
        }

        try {
            supabaseClient = window.supabase.createClient(config.supabaseUrl, config.publishableKey, {
                auth: {
                    persistSession: true,
                    autoRefreshToken: true,
                    detectSessionInUrl: true
                }
            });

            // Initialize Auth listener
            supabaseClient.auth.onAuthStateChange(async (event, session) => {
                console.log("[NutriScanAI Auth Event]:", event);
                currentUser = session ? session.user : null;
                updateAuthUI();

                if (event === 'SIGNED_IN' || event === 'INITIAL_SESSION') {
                    if (currentUser) {
                        await fetchCloudUserData();
                        await drainMutationQueue();
                        subscribeRealtime();
                    }
                } else if (event === 'SIGNED_OUT') {
                    unsubscribeRealtime();
                    updateAuthUI();
                }
            });

            return supabaseClient;
        } catch (err) {
            console.error("Failed to initialize Supabase client:", err);
            return null;
        }
    }

    function getClient() {
        if (!supabaseClient) {
            initClient();
        }
        return supabaseClient;
    }

    // ─────────────────────────────────────────────────────────────
    // 2. AUTHENTICATION MANAGER
    // ─────────────────────────────────────────────────────────────
    const AuthManager = {
        async signUp(email, password, metadata) {
            const client = getClient();
            if (!client) throw new Error("Supabase is not configured.");

            const metaObj = typeof metadata === 'string' ? { full_name: metadata } : (metadata || {});
            const { data, error } = await client.auth.signUp({
                email: email.trim(),
                password: password,
                options: {
                    data: {
                        full_name: (metaObj.full_name || '').trim() || 'NutriScan User',
                        dob: metaObj.dob || '',
                        age: metaObj.age || null,
                        height: metaObj.height || null,
                        weight: metaObj.weight || null
                    }
                }
            });

            if (error) throw error;
            currentUser = data.user;
            return data;
        },

        async signIn(email, password) {
            const client = getClient();
            if (!client) throw new Error("Supabase is not configured.");

            const { data, error } = await client.auth.signInWithPassword({
                email: email.trim(),
                password: password
            });

            if (error) throw error;
            currentUser = data.user;
            return data;
        },

        async signOut() {
            const client = getClient();
            if (client) {
                try {
                    await client.auth.signOut();
                } catch (e) {
                    console.warn("Sign out network error, clearing local session:", e);
                }
            }
            currentUser = null;
            localStorage.removeItem('nutriscan_user_mode');
            if (typeof window.checkLoginState === 'function') {
                window.checkLoginState();
            } else {
                updateAuthUI();
            }
            if (window.showToast) window.showToast("Signed out. Operating in Local Mode.");
        },

        async resetPassword(email) {
            const client = getClient();
            if (!client) throw new Error("Supabase is not configured.");

            const { data, error } = await client.auth.resetPasswordForEmail(email.trim(), {
                redirectTo: window.location.origin
            });
            if (error) throw error;
            return data;
        },

        getCurrentUser() {
            return currentUser;
        },

        isAuthenticated() {
            return currentUser !== null;
        }
    };

    // ─────────────────────────────────────────────────────────────
    // 3. IMAGE COMPRESSION & SUPABASE STORAGE PIPELINE
    // ─────────────────────────────────────────────────────────────
    async function compressPlateImage(fileOrDataUrl, maxDimension = 1280, quality = 0.85) {
        return new Promise((resolve) => {
            const img = new Image();
            img.crossOrigin = "anonymous";
            img.onload = () => {
                let width = img.width;
                let height = img.height;
                if (width > maxDimension || height > maxDimension) {
                    if (width > height) {
                        height = Math.round((height * maxDimension) / width);
                        width = maxDimension;
                    } else {
                        width = Math.round((width * maxDimension) / height);
                        height = maxDimension;
                    }
                }
                const canvas = document.createElement('canvas');
                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, width, height);

                const dataUrl = canvas.toDataURL('image/jpeg', quality);
                canvas.toBlob((blob) => {
                    resolve({ blob, dataUrl });
                }, 'image/jpeg', quality);
            };
            img.onerror = () => {
                resolve({ blob: null, dataUrl: typeof fileOrDataUrl === 'string' ? fileOrDataUrl : null });
            };
            img.src = typeof fileOrDataUrl === 'string' ? fileOrDataUrl : URL.createObjectURL(fileOrDataUrl);
        });
    }

    async function uploadPlateImageToStorage(blob, userId, mealId) {
        const client = getClient();
        if (!client || !userId || !blob) return null;

        try {
            const fileName = `${userId}/${mealId}_${Date.now()}.jpg`;
            const { data, error } = await client.storage
                .from('meal-images')
                .upload(fileName, blob, {
                    contentType: 'image/jpeg',
                    upsert: true
                });

            if (error) {
                console.warn("Storage upload note (will use local preview):", error.message);
                return null;
            }

            const { data: pubData } = client.storage
                .from('meal-images')
                .getPublicUrl(fileName);

            return pubData ? pubData.publicUrl : null;
        } catch (e) {
            console.warn("Plate image upload error:", e);
            return null;
        }
    }

    // ─────────────────────────────────────────────────────────────
    // 4. OFFLINE MUTATION QUEUE
    // ─────────────────────────────────────────────────────────────
    function getQueue() {
        try {
            const raw = localStorage.getItem(QUEUE_STORAGE_KEY);
            return raw ? JSON.parse(raw) : [];
        } catch (e) {
            return [];
        }
    }

    function enqueueMutation(op, data) {
        const queue = getQueue();
        queue.push({
            id: `mutation-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`,
            op,
            data,
            timestamp: Date.now()
        });
        try {
            localStorage.setItem(QUEUE_STORAGE_KEY, JSON.stringify(queue));
        } catch (e) {
            console.error("Queue store error:", e);
        }
        updateSyncPill();
    }

    async function drainMutationQueue() {
        if (!AuthManager.isAuthenticated()) return;
        const queue = getQueue();
        if (queue.length === 0) return;

        console.log(`[NutriScanAI Sync]: Draining ${queue.length} pending offline mutations...`);
        setSyncState(true);

        const remaining = [];
        for (const item of queue) {
            try {
                if (item.op === 'INSERT_MEAL') {
                    await sendMealToCloud(item.data);
                } else if (item.op === 'DELETE_MEAL') {
                    await deleteMealFromCloud(item.data.id);
                } else if (item.op === 'UPDATE_PROFILE') {
                    await sendProfileToCloud(item.data);
                } else if (item.op === 'UPDATE_GOALS') {
                    await sendGoalsToCloud(item.data);
                } else if (item.op === 'UPDATE_WATER') {
                    await sendWaterToCloud(item.data.date, item.data.ml);
                }
            } catch (err) {
                console.warn(`Retry needed for mutation ${item.id}:`, err);
                remaining.push(item);
            }
        }

        try {
            localStorage.setItem(QUEUE_STORAGE_KEY, JSON.stringify(remaining));
        } catch (e) {}

        setSyncState(false);
        updateSyncPill();
    }

    // ─────────────────────────────────────────────────────────────
    // 5. DATABASE CLOUD CRUD & SYNCHRONIZATION
    // ─────────────────────────────────────────────────────────────
    async function fetchCloudUserData() {
        const client = getClient();
        if (!client || !currentUser) return;

        setSyncState(true);
        try {
            const userId = currentUser.id;

            // A. Fetch Profile
            const { data: profileData } = await client
                .from('profiles')
                .select('*')
                .eq('id', userId)
                .single();

            if (profileData && window.NutriScanStorage) {
                const mappedProfile = {
                    name: profileData.full_name || 'Alex Morgan',
                    avatar: profileData.avatar_initial || 'A',
                    tagline: profileData.tagline || 'Personal nutrition space',
                    age: profileData.age || 28,
                    gender: profileData.gender || 'Female',
                    weight: parseFloat(profileData.weight_kg) || 68,
                    height: parseFloat(profileData.height_cm) || 175,
                    activity: profileData.activity_level || 'light',
                    goal: profileData.dietary_goal || 'Maintain weight & vitality'
                };
                window.NutriScanStorage.saveProfile(mappedProfile);
            }

            // B. Fetch Goals
            const { data: goalsData } = await client
                .from('goals')
                .select('*')
                .eq('user_id', userId)
                .single();

            if (goalsData && window.NutriScanStorage) {
                const mappedGoals = {
                    calories: goalsData.target_calories || 2000,
                    protein: parseFloat(goalsData.target_protein_g) || 100,
                    carbs: parseFloat(goalsData.target_carb_g) || 220,
                    fat: parseFloat(goalsData.target_fat_g) || 65,
                    water: parseFloat(goalsData.target_water_l) || 2.5,
                    dietaryPatterns: goalsData.dietary_patterns || ["Vegetarian", "High-Protein"],
                    allergens: goalsData.allergens || "Peanuts, Excess sodium",
                    schedule: goalsData.meal_schedule || "Breakfast: 9:00 AM · Lunch: 1:00 PM · Dinner: 7:30 PM"
                };
                window.NutriScanStorage.saveGoals(mappedGoals);
            }

            // C. Fetch Meals and their Components
            const { data: mealsData, error: mealsErr } = await client
                .from('meals')
                .select('*, meal_components(*)')
                .eq('user_id', userId)
                .order('created_at', { ascending: false });

            if (!mealsErr && Array.isArray(mealsData) && window.NutriScanStorage) {
                if (mealsData.length > 0) {
                    const mappedMeals = mealsData.map(m => ({
                        id: m.client_id || m.id,
                        cloudId: m.id,
                        name: m.name,
                        type: m.meal_type,
                        time: m.time_str,
                        date: m.log_date,
                        calories: m.calories,
                        protein: parseFloat(m.protein_g) || 0,
                        carbs: parseFloat(m.carb_g) || 0,
                        fat: parseFloat(m.fat_g) || 0,
                        fiber: parseFloat(m.fiber_g) || 0,
                        image: m.image_url || 'assets/nourish-bowl.png',
                        components: (m.meal_components || []).map(c => ({
                            name: c.name,
                            mass: parseFloat(c.mass_g) || 80,
                            calories: c.calories || 100,
                            p: parseFloat(c.protein_g) || 0,
                            c: parseFloat(c.carb_g) || 0,
                            f: parseFloat(c.fat_g) || 0
                        }))
                    }));
                    window.NutriScanStorage.saveMeals(mappedMeals);
                }
            }

            // D. Fetch Today's Water
            const todayStr = new Date().toISOString().split('T')[0];
            const { data: waterData } = await client
                .from('water_logs')
                .select('*')
                .eq('user_id', userId)
                .eq('log_date', todayStr)
                .single();

            if (waterData && window.NutriScanStorage) {
                window.NutriScanStorage.saveWater(waterData.amount_ml || 0);
            }

            if (window.nutriScanRefreshActiveView) {
                window.nutriScanRefreshActiveView();
            }

            console.log("[NutriScanAI Sync]: Cloud data successfully synchronized.");
        } catch (err) {
            console.error("Error fetching cloud data from Supabase:", err);
        } finally {
            setSyncState(false);
            updateSyncPill();
        }
    }

    async function sendMealToCloud(meal) {
        const client = getClient();
        if (!client || !currentUser) {
            enqueueMutation('INSERT_MEAL', meal);
            return;
        }

        try {
            // Check if meal photo needs compression and upload
            let cloudImageUrl = meal.image;
            if (meal.image && (meal.image.startsWith('blob:') || meal.image.startsWith('data:'))) {
                const compressed = await compressPlateImage(meal.image);
                if (compressed.blob) {
                    const uploadedUrl = await uploadPlateImageToStorage(compressed.blob, currentUser.id, meal.id);
                    if (uploadedUrl) cloudImageUrl = uploadedUrl;
                }
            }

            const mealPayload = {
                user_id: currentUser.id,
                client_id: meal.id,
                name: meal.name,
                meal_type: meal.type,
                time_str: meal.time,
                log_date: meal.date,
                calories: Math.round(meal.calories),
                protein_g: meal.protein || 0,
                carb_g: meal.carbs || 0,
                fat_g: meal.fat || 0,
                fiber_g: meal.fiber || 0,
                image_url: cloudImageUrl || 'assets/nourish-bowl.png'
            };

            const { data: insertedMeal, error: mealErr } = await client
                .from('meals')
                .insert(mealPayload)
                .select()
                .single();

            if (mealErr) throw mealErr;

            if (Array.isArray(meal.components) && meal.components.length > 0 && insertedMeal) {
                const componentsPayload = meal.components.map(c => ({
                    meal_id: insertedMeal.id,
                    name: c.name,
                    mass_g: c.mass || 80,
                    calories: Math.round(c.calories || 100),
                    protein_g: c.p || 0,
                    carb_g: c.c || 0,
                    fat_g: c.f || 0
                }));

                await client.from('meal_components').insert(componentsPayload);
            }
        } catch (err) {
            console.warn("Meal cloud sync failed, queued for retry:", err);
            enqueueMutation('INSERT_MEAL', meal);
        }
    }

    async function deleteMealFromCloud(mealId) {
        const client = getClient();
        if (!client || !currentUser) {
            enqueueMutation('DELETE_MEAL', { id: mealId });
            return;
        }

        try {
            // Delete matching either client_id or UUID
            await client
                .from('meals')
                .delete()
                .or(`client_id.eq.${mealId},id.eq.${mealId}`);
        } catch (err) {
            console.warn("Delete meal cloud sync failed, queued for retry:", err);
            enqueueMutation('DELETE_MEAL', { id: mealId });
        }
    }

    async function sendProfileToCloud(profile) {
        const client = getClient();
        if (!client || !currentUser) {
            enqueueMutation('UPDATE_PROFILE', profile);
            return;
        }

        try {
            await client.from('profiles').upsert({
                id: currentUser.id,
                full_name: profile.name,
                avatar_initial: profile.avatar || profile.name.charAt(0).toUpperCase(),
                tagline: profile.tagline,
                age: profile.age,
                gender: profile.gender,
                weight_kg: profile.weight,
                height_cm: profile.height,
                activity_level: profile.activity,
                dietary_goal: profile.goal,
                updated_at: new Date().toISOString()
            });
        } catch (err) {
            enqueueMutation('UPDATE_PROFILE', profile);
        }
    }

    async function sendGoalsToCloud(goals) {
        const client = getClient();
        if (!client || !currentUser) {
            enqueueMutation('UPDATE_GOALS', goals);
            return;
        }

        try {
            await client.from('goals').upsert({
                user_id: currentUser.id,
                target_calories: goals.calories,
                target_protein_g: goals.protein,
                target_carb_g: goals.carbs,
                target_fat_g: goals.fat,
                target_water_l: goals.water,
                dietary_patterns: goals.dietaryPatterns || [],
                allergens: goals.allergens || '',
                meal_schedule: goals.schedule || '',
                updated_at: new Date().toISOString()
            });
        } catch (err) {
            enqueueMutation('UPDATE_GOALS', goals);
        }
    }

    async function sendWaterToCloud(dateStr, ml) {
        const client = getClient();
        if (!client || !currentUser) {
            enqueueMutation('UPDATE_WATER', { date: dateStr, ml });
            return;
        }

        try {
            await client.from('water_logs').upsert({
                user_id: currentUser.id,
                log_date: dateStr,
                amount_ml: ml,
                updated_at: new Date().toISOString()
            }, { onConflict: 'user_id,log_date' });
        } catch (err) {
            enqueueMutation('UPDATE_WATER', { date: dateStr, ml });
        }
    }

    async function importLocalDataToCloud() {
        if (!currentUser || !window.NutriScanStorage) return;

        setSyncState(true);
        if (window.showToast) window.showToast("Importing local diary to your cloud account...");

        try {
            const localMeals = window.NutriScanStorage.getMeals();
            for (const m of localMeals) {
                await sendMealToCloud(m);
            }

            const localProfile = window.NutriScanStorage.getProfile();
            await sendProfileToCloud(localProfile);

            const localGoals = window.NutriScanStorage.getGoals();
            await sendGoalsToCloud(localGoals);

            const localWater = window.NutriScanStorage.getWater();
            await sendWaterToCloud(localWater.date, localWater.ml);

            if (window.showToast) window.showToast("✦ Local diary imported to cloud successfully!");
        } catch (e) {
            console.error("Local data import error:", e);
        } finally {
            setSyncState(false);
            updateSyncPill();
        }
    }

    // ─────────────────────────────────────────────────────────────
    // 6. REALTIME WEBSOCKET SUBSCRIPTION (MULTI-DEVICE LIVE SYNC)
    // ─────────────────────────────────────────────────────────────
    function subscribeRealtime() {
        const client = getClient();
        if (!client || !currentUser) return;
        unsubscribeRealtime();

        try {
            realtimeChannel = client
                .channel(`user-sync-${currentUser.id}`)
                .on('postgres_changes', { event: '*', schema: 'public', table: 'meals' }, (payload) => {
                    console.log("[Realtime Meal Change]:", payload);
                    fetchCloudUserData();
                })
                .on('postgres_changes', { event: '*', schema: 'public', table: 'water_logs' }, (payload) => {
                    console.log("[Realtime Water Change]:", payload);
                    fetchCloudUserData();
                })
                .subscribe();
        } catch (e) {
            console.warn("Realtime subscription note:", e);
        }
    }

    function unsubscribeRealtime() {
        if (realtimeChannel && supabaseClient) {
            supabaseClient.removeChannel(realtimeChannel);
            realtimeChannel = null;
        }
    }

    // ─────────────────────────────────────────────────────────────
    // 7. UI CONTROLLERS & STATUS BADGES
    // ─────────────────────────────────────────────────────────────
    function setSyncState(syncing) {
        isSyncing = syncing;
        updateSyncPill();
    }

    function updateSyncPill() {
        const pill = document.getElementById('cloud-sync-pill');
        const dot = document.getElementById('cloud-sync-dot');
        const text = document.getElementById('cloud-sync-text');
        if (!pill || !dot || !text) return;

        const queueCount = getQueue().length;

        if (isSyncing) {
            dot.className = 'status-dot dot-syncing';
            text.textContent = 'Syncing…';
        } else if (AuthManager.isAuthenticated()) {
            if (queueCount > 0) {
                dot.className = 'status-dot dot-pending';
                text.textContent = `Pending (${queueCount})`;
            } else {
                dot.className = 'status-dot dot-online';
                text.textContent = 'Cloud Synced';
            }
        } else {
            const hasLocal = localStorage.getItem('nutriscan_user_mode') === 'local' || (window.NutriScanStorage && window.NutriScanStorage.getProfile());
            if (hasLocal) {
                dot.className = 'status-dot dot-local';
                text.textContent = 'Local Mode';
            } else {
                dot.className = 'status-dot dot-guest';
                text.textContent = 'Setup Required';
            }
        }
    }

    function updateAuthUI() {
        const topbarName = document.getElementById('topbar-name');
        const topbarAvatar = document.getElementById('topbar-avatar');
        const sideAvatar = document.getElementById('side-avatar');
        const sideName = document.getElementById('side-name');
        const sideTagline = document.getElementById('side-tagline');

        const profileEmail = document.getElementById('profile-account-email');
        const profileStatus = document.getElementById('profile-account-status');
        const btnAuthAction = document.getElementById('btn-profile-auth-action');

        if (currentUser) {
            const name = currentUser.user_metadata?.full_name || currentUser.email.split('@')[0];
            const initial = (name.charAt(0) || 'U').toUpperCase();

            if (topbarName) topbarName.textContent = name.split(' ')[0];
            if (topbarAvatar) topbarAvatar.textContent = initial;
            if (sideAvatar) sideAvatar.textContent = initial;
            if (sideName) sideName.textContent = name;
            if (sideTagline) sideTagline.textContent = currentUser.email;

            if (profileEmail) profileEmail.textContent = currentUser.email;
            if (profileStatus) profileStatus.innerHTML = '<span class="badge" style="background:var(--forest);color:#fff">Cloud Connected</span>';
            if (btnAuthAction) {
                btnAuthAction.textContent = 'Sign Out';
                btnAuthAction.onclick = () => AuthManager.signOut();
            }
        } else {
            const localProfile = (window.NutriScanStorage && typeof window.NutriScanStorage.getProfile === 'function') 
                ? window.NutriScanStorage.getProfile() 
                : null;

            if (localProfile && localProfile.name) {
                const name = localProfile.name;
                const initial = (name.charAt(0) || 'U').toUpperCase();

                if (topbarName) topbarName.textContent = name.split(' ')[0];
                if (topbarAvatar) topbarAvatar.textContent = initial;
                if (sideAvatar) sideAvatar.textContent = initial;
                if (sideName) sideName.textContent = name;
                if (sideTagline) sideTagline.textContent = 'Local Offline Mode';

                if (profileEmail) profileEmail.textContent = 'Local Offline Storage';
                if (profileStatus) profileStatus.innerHTML = '<span class="badge" style="background:var(--sage-soft);color:var(--forest);font-weight:700">● Local Mode</span>';
                if (btnAuthAction) {
                    btnAuthAction.textContent = 'Connect Cloud Account';
                    btnAuthAction.onclick = () => window.nutriScanOpenAuthModal('signup', false);
                }
            } else {
                if (topbarName) topbarName.textContent = 'Guest';
                if (topbarAvatar) topbarAvatar.textContent = '?';
                if (sideAvatar) sideAvatar.textContent = '?';
                if (sideName) sideName.textContent = 'NutriScan User';
                if (sideTagline) sideTagline.textContent = 'Setup required';

                if (profileEmail) profileEmail.textContent = 'Not configured';
                if (profileStatus) profileStatus.innerHTML = '<span class="badge">Unauthenticated</span>';
                if (btnAuthAction) {
                    btnAuthAction.textContent = 'Sign In / Local Setup';
                    btnAuthAction.onclick = () => window.nutriScanOpenAuthModal('signup', true);
                }
            }
        }

        updateSyncPill();
    }

    // Auto-listen to window online events to drain queues
    window.addEventListener('online', () => {
        console.log("[NutriScanAI]: Network connection restored.");
        drainMutationQueue();
    });

    // Expose Sync API onto window
    window.NutriScanSync = {
        init: initClient,
        getClient: getClient,
        auth: AuthManager,
        fetchCloudUserData,
        sendMealToCloud,
        deleteMealFromCloud,
        sendProfileToCloud,
        sendGoalsToCloud,
        sendWaterToCloud,
        importLocalDataToCloud,
        compressPlateImage,
        drainMutationQueue,
        updateAuthUI,
        updateSyncPill
    };

    // Auto-initialize when script loads
    document.addEventListener('DOMContentLoaded', () => {
        initClient();
    });

})();
