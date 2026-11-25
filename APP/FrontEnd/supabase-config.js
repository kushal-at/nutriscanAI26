/**
 * NutriScanAI — Supabase Cloud Configuration
 * Manages project credentials, API URLs, and user overrides.
 */

(function () {
    'use strict';

    const DEFAULT_CONFIG = {
        projectId: "jndsncexhhabcibtzxts",
        projectRegion: "ap-northeast-1",
        supabaseUrl: "https://jndsncexhhabcibtzxts.supabase.co",
        publishableKey: "sb_publishable_vwDsYMPd45aFbFOM0SbGkA_XN9nWyiV",
        storageBucket: "meal-images"
    };

    const STORAGE_KEY_CONFIG = 'nutriscan_supabase_config';

    function getSupabaseConfig() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY_CONFIG);
            if (raw) {
                const parsed = JSON.parse(raw);
                return { ...DEFAULT_CONFIG, ...parsed };
            }
        } catch (e) {
            console.warn("Could not read custom Supabase config from localStorage:", e);
        }
        return { ...DEFAULT_CONFIG };
    }

    function saveSupabaseConfig(url, key) {
        try {
            const current = getSupabaseConfig();
            current.supabaseUrl = (url || '').trim().replace(/\/+$/, '');
            current.publishableKey = (key || '').trim();
            localStorage.setItem(STORAGE_KEY_CONFIG, JSON.stringify(current));
            return true;
        } catch (e) {
            console.error("Could not save custom Supabase config:", e);
            return false;
        }
    }

    function resetSupabaseConfig() {
        try {
            localStorage.removeItem(STORAGE_KEY_CONFIG);
            return true;
        } catch (e) {
            return false;
        }
    }

    window.NutriScanSupabaseConfig = {
        get: getSupabaseConfig,
        save: saveSupabaseConfig,
        reset: resetSupabaseConfig,
        defaults: DEFAULT_CONFIG
    };

})();
