"use client";
// Chart primitives. Palette: validated dark categorical slots (fixed order, never cycled).
// Thin 2px lines, rounded bar ends, recessive grid, hover tooltips, legend when >= 2 series.
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)", "var(--series-5)", "var(--series-6)", "var(--series-7)", "var(--series-8)"];
const axis = { stroke: "#4a4a4a", tick: { fill: "#8b8a83", fontSize: 11 }, tickLine: false };
const tip = { contentStyle: { background: "#1b1b1b", border: "1px solid #343434", borderRadius: 8, fontSize: 12, color: "#f2f1ee" }, labelStyle: { color: "#c3c2b7" }, itemStyle: { color: "#f2f1ee" }, cursor: { fill: "rgba(255,255,255,.04)" } };

export function HBar({ data, x, y, height = 260, fmt, colorBy }: { data: any[]; x: string; y: string; height?: number; fmt?: (v: number) => string; colorBy?: (d: any, i: number) => string }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24, top: 4, bottom: 4 }} barCategoryGap={4}>
        <CartesianGrid horizontal={false} stroke="#262626" />
        <XAxis type="number" {...axis} tickFormatter={fmt} />
        <YAxis type="category" dataKey={y} {...axis} width={150} />
        <Tooltip {...tip} formatter={(v: any) => (fmt ? fmt(Number(v)) : v)} />
        <Bar dataKey={x} radius={[0, 4, 4, 0]} fill={SERIES[0]}>
          {colorBy && data.map((d, i) => <Cell key={i} fill={colorBy(d, i)} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function VBars({ data, x, series, height = 260, fmt, names }: { data: any[]; x: string; series: string[]; height?: number; fmt?: (v: number) => string; names?: Record<string, string> }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ left: 0, right: 8, top: 8, bottom: 4 }} barGap={2}>
        <CartesianGrid vertical={false} stroke="#262626" />
        <XAxis dataKey={x} {...axis} interval={0} />
        <YAxis {...axis} tickFormatter={fmt} width={48} />
        <Tooltip {...tip} formatter={(v: any, n: any) => [fmt ? fmt(Number(v)) : v, names?.[n] ?? n]} />
        {series.length > 1 && <Legend wrapperStyle={{ fontSize: 12, color: "#c3c2b7" }} formatter={(n) => names?.[n] ?? n} />}
        {series.map((s, i) => <Bar key={s} dataKey={s} fill={SERIES[i]} radius={[4, 4, 0, 0]} />)}
      </BarChart>
    </ResponsiveContainer>
  );
}

export function Lines({ data, x, series, height = 260, fmt, names, ref0 }: { data: any[]; x: string; series: string[]; height?: number; fmt?: (v: number) => string; names?: Record<string, string>; ref0?: boolean }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ left: 0, right: 12, top: 8, bottom: 4 }}>
        <CartesianGrid vertical={false} stroke="#262626" />
        <XAxis dataKey={x} {...axis} />
        <YAxis {...axis} tickFormatter={fmt} width={52} />
        <Tooltip {...tip} cursor={{ stroke: "#4a4a4a" }} formatter={(v: any, n: any) => [fmt ? fmt(Number(v)) : v, names?.[n] ?? n]} />
        {series.length > 1 && <Legend wrapperStyle={{ fontSize: 12, color: "#c3c2b7" }} formatter={(n) => names?.[n] ?? n} />}
        {ref0 && <ReferenceLine y={0} stroke="#6b6b6b" />}
        {series.map((s, i) => <Line key={s} dataKey={s} stroke={SERIES[i]} strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} isAnimationActive={false} />)}
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Stacked share-over-time (taste evolution). Genres keep a stable color by first appearance. */
export function StackedShare({ data, keys, height = 280 }: { data: any[]; keys: string[]; height?: number }) {
  const shown = keys.slice(0, 7);
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ left: 0, right: 12, top: 8, bottom: 4 }} stackOffset="expand">
        <CartesianGrid vertical={false} stroke="#262626" />
        <XAxis dataKey="date" {...axis} />
        <YAxis {...axis} tickFormatter={(v) => `${Math.round(v * 100)}%`} width={44} />
        <Tooltip {...tip} cursor={{ stroke: "#4a4a4a" }} formatter={(v: any, n: any) => [`${(100 * Number(v)).toFixed(0)}%`, n]} />
        <Legend wrapperStyle={{ fontSize: 12, color: "#c3c2b7" }} />
        {shown.map((k, i) => <Area key={k} type="monotone" dataKey={k} stackId="1" stroke="#141414" strokeWidth={2} fill={SERIES[i]} fillOpacity={0.9} isAnimationActive={false} />)}
      </AreaChart>
    </ResponsiveContainer>
  );
}
