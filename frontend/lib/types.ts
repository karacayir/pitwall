// Mirrors backend/app/schemas.py — the WebSocket / REST contracts.

export interface Quantiles {
  p10: number;
  p50: number;
  p90: number;
}

export interface Forecast {
  current: Quantiles | null;
  fresh: Record<string, Quantiles>;
  ahead: Record<string, number[]>; // p50 curves; keys: "current", "SOFT", ...
}

export interface WeatherInfo {
  track_temp: number | null;
  air_temp: number | null;
  humidity: number | null;
  rain: boolean;
  rain_prob_15m: number | null;
}

export interface DriverUpdate {
  driver_number: number;
  driver_code: string | null;
  team_id: string | null;
  position: number | null;
  gap_ahead_s: number | null;
  compound: string | null;
  tyre_age: number | null;
  stint_no: number | null;
  last_lap_s: number | null;
  lap_number: number | null;
  pitted: boolean;
  bias_s: number | null;
  forecast: Forecast | null;
}

export interface LapUpdate {
  type: "lap_update";
  session_key: string;
  lap: number;
  laps_total: number;
  track_status: "green" | "sc" | "vsc" | "yellow" | "red";
  reference_pace_s: number | null;
  weather: WeatherInfo;
  drivers: DriverUpdate[];
  _id?: number;
}

export interface SessionInfo {
  session_id: string;
  event_name: string | null;
  track_id: string;
  laps_total: number;
  data_source: "replay" | "live";
  replay_speed: number | null;
}

export interface StintPlan {
  lap: number;
  compound: string;
}

export interface StrategyResult {
  stops: StintPlan[];
  label: string;
  p_position: Record<string, number>;
  finish_time_s: Quantiles;
  expected_position: number;
  p_better_than_current: number | null;
}

export interface SimulateResponse {
  driver_number: number;
  from_lap: number;
  n_sims: number;
  baseline: StrategyResult;
  strategies: StrategyResult[];
}

export const COMPOUND_COLORS: Record<string, string> = {
  SOFT: "#F0453F",
  MEDIUM: "#F7C325",
  HARD: "#E8E8EC",
  INTERMEDIATE: "#3DBE5B",
  WET: "#2E8BE0",
};

export const COMPOUND_SHORT: Record<string, string> = {
  SOFT: "S",
  MEDIUM: "M",
  HARD: "H",
  INTERMEDIATE: "I",
  WET: "W",
};
