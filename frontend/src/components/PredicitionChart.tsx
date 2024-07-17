import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
const chartConfig = {
  power: {
    label: "Power (kw)",
    color: "hsl(var(--chart-1))",
  },
} satisfies ChartConfig;

function CustomizedTick(props) {
  const { x, y, stroke, payload } = props;
  const [date, time] = payload.value.split("T");
  return (
    <g transform={`translate(${x},${y})`}>
      <text x={0} y={0} dy={16}>
        <tspan textAnchor="middle" x="0">
          {time.slice(0, 5)}
        </tspan>
        <tspan textAnchor="middle" x="0" dy="20">
          {date}
        </tspan>
      </text>
    </g>
  );
}

export function PredictionChart({ predictions }) {
  const chartData = Object.keys(predictions).map((key) => ({
    datetime: key,
    power: predictions[key],
  }));

  return (
    <ChartContainer config={chartConfig}>
      <LineChart accessibilityLayer data={chartData} margin={{ bottom: 40 }}>
        <CartesianGrid vertical={false} />
        <XAxis
          dataKey="datetime"
          axisLine={false}
          tickLine={false}
          tick={<CustomizedTick />}
        />
        <YAxis
          dataKey="power"
          axisLine={false}
          tickLine={false}
          tickMargin={8}
          label={{
            value: "Power (kw)",
            angle: -90,
            position: "insideLeft",
          }}
        />
        <ChartTooltip cursor={false} />
        <Line
          dataKey="power"
          type="natural"
          stroke="var(--color-power)"
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ChartContainer>
  );
}
