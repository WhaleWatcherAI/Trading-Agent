export type SideTxt = 'Buy' | 'Sell';

export class PositionTracker {
  position = 0;
  avgPrice = 0;
  realized = 0;

  constructor(
    private multiplier = Number(process.env.TOPSTEPX_MULTIPLIER ?? 5),
    private commissionPerSide = Number(process.env.TOPSTEPX_COMMISSION_PER_SIDE ?? 0)
  ) {}

  onFill(side: SideTxt, qty: number, price: number) {
    const signed = side === 'Buy' ? qty : -qty;

    if ((this.position >= 0 && signed > 0) || (this.position <= 0 && signed < 0)) {
      const newPos = this.position + signed;
      if (newPos === 0) {
        const pnlPer = (price - this.avgPrice) * Math.sign(signed);
        this.realized += pnlPer * Math.abs(signed) * this.multiplier - this.commissionPerSide;
        this.position = 0;
        this.avgPrice = 0;
      } else {
        const notional = this.avgPrice * Math.abs(this.position) + price * Math.abs(signed);
        this.position = newPos;
        this.avgPrice = notional / Math.abs(this.position);
      }
      return;
    }

    const closingQty = Math.min(Math.abs(this.position), Math.abs(signed)) * Math.sign(signed);
    const pnlPer = (price - this.avgPrice) * Math.sign(this.position);
    this.realized += pnlPer * Math.abs(closingQty) * this.multiplier - this.commissionPerSide;

    const newPos = this.position + signed;
    if (newPos === 0) {
      this.position = 0;
      this.avgPrice = 0;
    } else if (Math.sign(newPos) !== Math.sign(this.position)) {
      this.position = newPos;
      this.avgPrice = price;
    } else {
      this.position = newPos;
    }
  }
}
