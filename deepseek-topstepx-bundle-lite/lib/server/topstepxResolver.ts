import { fetchTopstepXFuturesMetadata } from '@/lib/topstepx';

let cachedContract: { id: string; name?: string } | null = null;

export async function resolveTopstepxContractId() {
  if (cachedContract) {
    return cachedContract.id;
  }

  const lookup =
    process.env.TOPSTEPX_SECOND_SMA_CONTRACT_ID ||
    process.env.TOPSTEPX_CONTRACT_ID ||
    process.env.TOPSTEPX_SECOND_SMA_SYMBOL ||
    process.env.TOPSTEPX_SMA_SYMBOL;

  if (!lookup) {
    throw new Error('TopstepX contract or symbol is not configured.');
  }

  const metadata = await fetchTopstepXFuturesMetadata(lookup);
  if (!metadata) {
    throw new Error(`Unable to resolve TopstepX metadata for ${lookup}`);
  }
  cachedContract = { id: metadata.id, name: metadata.name };
  return metadata.id;
}
