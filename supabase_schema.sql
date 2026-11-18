-- ============================================================================
-- NutriScanAI — Production Supabase PostgreSQL Schema Migration
-- Database: PostgreSQL 15+ (Supabase)
-- ============================================================================

-- 0. Extensions
create extension if not exists "uuid-ossp";

-- 1. PROFILES TABLE
create table if not exists public.profiles (
    id uuid primary key references auth.users(id) on delete cascade,
    full_name text not null default 'Alex Morgan',
    avatar_initial text default 'A',
    tagline text default 'Personal nutrition space',
    age integer default 28 check (age between 5 and 120),
    gender text default 'Female',
    weight_kg numeric default 68.0 check (weight_kg > 0),
    height_cm numeric default 175.0 check (height_cm > 0),
    activity_level text default 'light' check (activity_level in ('sedentary', 'light', 'moderate', 'very', 'extra')),
    dietary_goal text default 'Maintain weight & vitality',
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 2. GOALS TABLE
create table if not exists public.goals (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references auth.users(id) on delete cascade not null unique,
    target_calories integer default 2000 check (target_calories >= 500),
    target_protein_g numeric default 100.0 check (target_protein_g >= 0),
    target_carb_g numeric default 220.0 check (target_carb_g >= 0),
    target_fat_g numeric default 65.0 check (target_fat_g >= 0),
    target_water_l numeric default 2.5 check (target_water_l >= 0),
    dietary_patterns text[] default array['Vegetarian', 'High-Protein'],
    allergens text default 'Peanuts, Excess sodium',
    meal_schedule text default 'Breakfast: 9:00 AM · Lunch: 1:00 PM · Dinner: 7:30 PM',
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 3. USER SETTINGS TABLE
create table if not exists public.user_settings (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references auth.users(id) on delete cascade not null unique,
    theme_preference text default 'light' check (theme_preference in ('light', 'dark', 'system')),
    measurement_system text default 'metric' check (measurement_system in ('metric', 'imperial')),
    auto_save_scans boolean default false,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 4. DAILY WATER LOGS TABLE
create table if not exists public.water_logs (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references auth.users(id) on delete cascade not null,
    log_date date not null,
    amount_ml integer default 0 check (amount_ml >= 0),
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null,
    unique(user_id, log_date)
);

-- 5. MEALS TABLE
create table if not exists public.meals (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references auth.users(id) on delete cascade not null,
    client_id text, -- Preserves local client-generated ID for offline reconciliation
    name text not null,
    meal_type text not null check (meal_type in ('Breakfast', 'Lunch', 'Dinner', 'Snack')),
    time_str text not null,
    log_date date not null,
    calories integer not null check (calories >= 0),
    protein_g numeric not null default 0 check (protein_g >= 0),
    carb_g numeric not null default 0 check (carb_g >= 0),
    fat_g numeric not null default 0 check (fat_g >= 0),
    fiber_g numeric not null default 0 check (fiber_g >= 0),
    plate_diameter_cm numeric default 25.0,
    scale_calibrated boolean default true,
    image_url text,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 6. MEAL COMPONENTS (INGREDIENTS) TABLE
create table if not exists public.meal_components (
    id uuid primary key default gen_random_uuid(),
    meal_id uuid references public.meals(id) on delete cascade not null,
    name text not null,
    mass_g numeric not null check (mass_g > 0),
    calories integer not null check (calories >= 0),
    protein_g numeric default 0 check (protein_g >= 0),
    carb_g numeric default 0 check (carb_g >= 0),
    fat_g numeric default 0 check (fat_g >= 0),
    fiber_g numeric default 0 check (fiber_g >= 0)
);

-- Performance Indexes
create index if not exists idx_meals_user_date on public.meals(user_id, log_date desc);
create index if not exists idx_water_user_date on public.water_logs(user_id, log_date desc);
create index if not exists idx_components_meal_id on public.meal_components(meal_id);

-- 7. AGGREGATE DAILY REPORT VIEW
create or replace view public.v_daily_nutrition_summary with (security_invoker = true) as
select 
    m.user_id,
    m.log_date,
    count(m.id) as meal_count,
    sum(m.calories) as total_calories,
    sum(m.protein_g) as total_protein_g,
    sum(m.carb_g) as total_carb_g,
    sum(m.fat_g) as total_fat_g,
    sum(m.fiber_g) as total_fiber_g,
    coalesce(w.amount_ml, 0) as water_intake_ml
from public.meals m
left join public.water_logs w on m.user_id = w.user_id and m.log_date = w.log_date
group by m.user_id, m.log_date, w.amount_ml;

-- 8. AUTOMATIC PROFILE & GOALS PROVISIONING TRIGGER
create or replace function public.handle_new_user() 
returns trigger as $$
begin
    -- Create default profile
    insert into public.profiles (id, full_name, avatar_initial)
    values (
        new.id,
        coalesce(new.raw_user_meta_data->>'full_name', split_part(new.email, '@', 1)),
        upper(substring(coalesce(new.raw_user_meta_data->>'full_name', split_part(new.email, '@', 1)) from 1 for 1))
    ) on conflict (id) do nothing;

    -- Create default goals
    insert into public.goals (user_id)
    values (new.id)
    on conflict (user_id) do nothing;

    -- Create default settings
    insert into public.user_settings (user_id)
    values (new.id)
    on conflict (user_id) do nothing;

    return new;
end;
$$ language plpgsql security definer;

-- Trigger execution on auth.users insert
drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
    after insert on auth.users
    for each row execute function public.handle_new_user();

-- 9. ROW-LEVEL SECURITY (RLS) POLICIES
alter table public.profiles enable row level security;
alter table public.goals enable row level security;
alter table public.user_settings enable row level security;
alter table public.water_logs enable row level security;
alter table public.meals enable row level security;
alter table public.meal_components enable row level security;

-- Profiles Policies
drop policy if exists "Users view own profile" on public.profiles;
create policy "Users view own profile" on public.profiles for select using (auth.uid() = id);

drop policy if exists "Users update own profile" on public.profiles;
create policy "Users update own profile" on public.profiles for update using (auth.uid() = id);

drop policy if exists "Users insert own profile" on public.profiles;
create policy "Users insert own profile" on public.profiles for insert with check (auth.uid() = id);

-- Goals & Settings Policies
drop policy if exists "Users manage own goals" on public.goals;
create policy "Users manage own goals" on public.goals for all using (auth.uid() = user_id);

drop policy if exists "Users manage own settings" on public.user_settings;
create policy "Users manage own settings" on public.user_settings for all using (auth.uid() = user_id);

drop policy if exists "Users manage own water_logs" on public.water_logs;
create policy "Users manage own water_logs" on public.water_logs for all using (auth.uid() = user_id);

-- Meals & Components Policies
drop policy if exists "Users manage own meals" on public.meals;
create policy "Users manage own meals" on public.meals for all using (auth.uid() = user_id);

drop policy if exists "Users manage own meal components" on public.meal_components;
create policy "Users manage own meal components" on public.meal_components
    for all using (
        exists (
            select 1 from public.meals 
            where meals.id = meal_components.meal_id 
            and meals.user_id = auth.uid()
        )
    );

-- 10. STORAGE BUCKET FOR PLATE PHOTOS
insert into storage.buckets (id, name, public)
values ('meal-images', 'meal-images', true)
on conflict (id) do nothing;

drop policy if exists "Authenticated users upload meal photos" on storage.objects;
create policy "Authenticated users upload meal photos" on storage.objects
    for insert with check (bucket_id = 'meal-images' and auth.role() = 'authenticated');

drop policy if exists "Public read meal photos" on storage.objects;
create policy "Public read meal photos" on storage.objects
    for select using (bucket_id = 'meal-images');

drop policy if exists "Users delete their meal photos" on storage.objects;
create policy "Users delete their meal photos" on storage.objects
    for delete using (bucket_id = 'meal-images' and auth.uid()::text = (storage.foldername(name))[1]);
