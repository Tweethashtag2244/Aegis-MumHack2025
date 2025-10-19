"use client";

import type { FC } from "react";
import { RadialBarChart, RadialBar, PolarAngleAxis } from "recharts";
import { ChartContainer } from "@/components/ui/chart";

interface ScoreGaugeProps {
  value: number;
}

export const ScoreGauge: FC<ScoreGaugeProps> = ({ value }) => {
  const getFillColor = (v: number) => {
    if (v > 70) return "hsl(var(--primary))";
    if (v > 40) return "hsl(var(--accent))";
    return "hsl(var(--destructive))";
  };

  const fillColor = getFillColor(value);

  const chartData = [{ name: "score", value, fill: fillColor }];

  return (
    <ChartContainer
      config={{
        score: {
          label: "Score",
        },
      }}
      className="mx-auto aspect-square w-full max-w-[200px]"
    >
      <RadialBarChart
        data={chartData}
        startAngle={-225}
        endAngle={45}
        innerRadius="80%"
        outerRadius="110%"
        barSize={20}
        cy="50%"
      >
        <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
        <RadialBar
          background
          dataKey="value"
          cornerRadius={10}
        />
        <text
          x="50%"
          y="50%"
          textAnchor="middle"
          dominantBaseline="middle"
          className="fill-foreground text-4xl font-bold"
        >
          {value.toFixed(0)}%
        </text>
        <text
          x="50%"
          y="68%"
          textAnchor="middle"
          dominantBaseline="middle"
          className="fill-muted-foreground text-sm"
        >
          Authenticity
        </text>
      </RadialBarChart>
    </ChartContainer>
  );
};
