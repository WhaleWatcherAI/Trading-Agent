const TOPSTEPX_USERNAME = process.env.TOPSTEPX_USERNAME;
const TOPSTEPX_API_KEY = process.env.TOPSTEPX_API_KEY;
const TOPSTEPX_BASE_URL = "https://api.topstepx.com";

interface Bar {
  t: string;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
}

async function authenticate(): Promise<string> {
  const response = await fetch(`${TOPSTEPX_BASE_URL}/api/Auth/loginKey`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      userName: TOPSTEPX_USERNAME,
      apiKey: TOPSTEPX_API_KEY,
    }),
  });
  const data = await response.json();
  if (!data.token) throw new Error("Auth failed");
  return data.token;
}

async function fetchBars(
  token: string,
  contractId: string,
  startTime: Date,
  endTime: Date
): Promise<Bar[]> {
  const payload = {
    contractId,
    live: false,
    startTime: startTime.toISOString(),
    endTime: endTime.toISOString(),
    unit: 1,
    unitNumber: 1,
  };

  const response = await fetch(`${TOPSTEPX_BASE_URL}/api/History/retrieveBars`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  return data.bars || [];
}

async function main() {
  const token = await authenticate();
  console.log("Authenticated");

  const symbols = [
    { name: "ES", contractId: "CON.F.US.EP.Z25" },
    { name: "NQ", contractId: "CON.F.US.ENQ.Z25" },
  ];

  const endTime = new Date();
  const startTime = new Date(endTime.getTime() - 24 * 60 * 60 * 1000);

  for (const sym of symbols) {
    console.log(`\nFetching ${sym.name}...`);
    let allBars: Bar[] = [];

    let chunkStart = new Date(startTime);
    while (chunkStart < endTime) {
      const chunkEnd = new Date(
        Math.min(chunkStart.getTime() + 4 * 60 * 60 * 1000, endTime.getTime())
      );

      const bars = await fetchBars(token, sym.contractId, chunkStart, chunkEnd);
      const startStr = chunkStart.toISOString();
      const endStr = chunkEnd.toISOString();
      console.log(`  Chunk ${startStr} -> ${endStr}: ${bars.length} bars`);
      allBars = allBars.concat(bars);

      chunkStart = chunkEnd;
      await new Promise((r) => setTimeout(r, 500));
    }

    const fs = await import("fs");
    const output = {
      symbol: sym.name,
      contractId: sym.contractId,
      bars: allBars,
    };
    fs.writeFileSync(
      `ml/data/bars_1s_${sym.name.toLowerCase()}_today.json`,
      JSON.stringify(output)
    );
    console.log(`Saved ${allBars.length} bars for ${sym.name}`);
  }
}

main().catch(console.error);
