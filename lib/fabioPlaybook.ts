export type MarketState =
  | 'balanced'
  | 'out_of_balance_uptrend'
  | 'out_of_balance_downtrend'
  | 'balanced_with_failed_breakout_above'
  | 'balanced_with_failed_breakout_below';

export type SessionName = 'London' | 'NewYork' | 'Other';

export type SetupModel = 'trend_continuation' | 'mean_reversion';

export interface RiskRules {
  riskPerTradePctMin: number;
  riskPerTradePctMax: number;
  stopPlacementDescription: string;
  breakEvenRuleDescription: string;
  invalidationDescription: string;
  defaultTargetDescription: string;
}

export interface VolumeProfileLevel {
  type: 'POC' | 'VAH' | 'VAL' | 'LVN';
  label: string;
  description: string;
}

export interface OrderFlowSignals {
  usesBigPrints: boolean;
  usesFootprintImbalance: boolean;
  usesCvd: boolean;
  aggressionDescription: string;
}

export interface SetupDefinition {
  id: SetupModel;
  name: string;
  marketStateFilter: string;
  sessionPreference: SessionName[];
  description: string;
  steps: string[];
  locationLogic: string;
  orderFlowLogic: string;
  riskManagementNotes: string[];
  targetLogic: string;
}

export interface ExampleTrade {
  exampleId: string;
  model: SetupModel;
  session: SessionName;
  marketState: MarketState;
  context: string;
  entryLogic: string;
  riskManagement: string;
  exitLogic: string;
}

export interface FabioPlaybookSpec {
  philosophy: string;
  decisionLayers: string[];
  instruments: string[];
  style: string;
  riskRules: RiskRules;
  volumeProfileConcepts: VolumeProfileLevel[];
  orderFlowSignals: OrderFlowSignals;
  setups: SetupDefinition[];
  examples: ExampleTrade[];
  metaPros: string[];
  metaCons: string[];
}

export const fabioPlaybook: FabioPlaybookSpec = {
  philosophy:
    'The market is an auction oscillating between balance (rotation around fair value) and imbalance (directional discovery away from prior value). The agent only trades when market state, location relative to volume-profile structure, and order-flow aggression all align.',
  decisionLayers: ['market_state', 'location', 'aggression'],
  instruments: ['Futures: NASDAQ, ES and similar index futures'],
  style: 'Intraday scalping, primarily during liquid session hours.',
  riskRules: {
    riskPerTradePctMin: 0.25,
    riskPerTradePctMax: 0.5,
    stopPlacementDescription:
      'Stops are placed just beyond the aggressive print, footprint, or zone that triggered the trade, with an additional 1–2 tick buffer beyond the obvious swing high or low.',
    breakEvenRuleDescription:
      'When CVD and tape confirm strong pressure in trade direction, move the stop to break-even earlier than usual to protect capital while keeping room for continuation.',
    invalidationDescription:
      'If the trade is wrong, it should become clear quickly. The stop is never widened; holding beyond invalidation zones is not allowed.',
    defaultTargetDescription:
      'The default profit target for both setups is the relevant balance POC. Extending beyond POC toward range extremes is only considered in exceptional trend conditions.'
  },
  volumeProfileConcepts: [
    {
      type: 'POC',
      label: 'Point of Control',
      description:
        'Price level with highest traded volume in the chosen profile; represents fair value and the center of balance.'
    },
    {
      type: 'VAH',
      label: 'Value Area High',
      description:
        'Upper boundary of the value area that contains approximately 70% of traded volume around the POC.'
    },
    {
      type: 'VAL',
      label: 'Value Area Low',
      description:
        'Lower boundary of the value area that contains approximately 70% of traded volume around the POC.'
    },
    {
      type: 'LVN',
      label: 'Low Volume Node',
      description:
        'Price level or small zone with noticeably lower volume relative to surrounding prices on the profile. Often corresponds to swift moves and serves as a primary reaction zone for entries inside impulse or reclaim legs.'
    }
  ],
  orderFlowSignals: {
    usesBigPrints: true,
    usesFootprintImbalance: true,
    usesCvd: true,
    aggressionDescription:
      'Aggression is defined as market orders hitting the book with clear directional bias, evidenced by unusually large prints at key levels, footprint imbalances, and a CVD slope aligned with the intended trade direction.'
  },
  setups: [
    {
      id: 'trend_continuation',
      name: 'Trend Model – Out-of-Balance to New Balance',
      marketStateFilter:
        'Out-of-balance state with directional displacement away from prior value; strong impulse leg present.',
      sessionPreference: ['NewYork'],
      description:
        'Continuation entries in the direction of a strong impulse leg that has broken prior balance. The agent uses volume profile on the impulse to locate LVNs and requires confirming buy or sell aggression on retest before entering.',
      steps: [
        'Confirm market is out-of-balance and trending away from prior value area and POC.',
        'Identify the latest impulse leg that broke structure and is clearly directional.',
        'Build a volume profile on that impulse leg and mark LVNs as potential reaction zones.',
        'Wait for price to retest an LVN in the direction of the main trend.',
        'At the LVN, require directional aggression (big prints, footprint imbalance, CVD confirmation) before entering.',
        'Place the stop just beyond the aggressive prints with a small tick buffer beyond the local swing.',
        'Target the POC of the next balance area the auction is likely to reach; optionally trail on rare strong trend days.'
      ],
      locationLogic:
        'Location is defined by LVNs inside the impulse leg profile. Entries are only considered when price pulls back into these LVNs in the direction of the prevailing trend.',
      orderFlowLogic:
        'On LVN retest, the tape should show clear directional aggression: large trades and/or footprint imbalances in trade direction, with CVD confirming sustained pressure. No aggression at the LVN means no trade.',
      riskManagementNotes: [
        'Stops are tight and placed beyond the aggressive footprint plus 1–2 ticks.',
        'If CVD bends strongly in favor of the trade, the stop can be moved to break-even earlier.',
        'No widening of stops; if structure fails, the trade is invalid.'
      ],
      targetLogic:
        'Base target is the POC of the balance the market is moving toward. This is where roughly 70% of rotational activity is expected; most of the position is closed there by default.'
    },
    {
      id: 'mean_reversion',
      name: 'Mean Reversion Model – Failed Breakout Back Into Balance',
      marketStateFilter:
        'Balanced state with a clearly defined value area, followed by one or more failed attempts to trade outside VAH or VAL.',
      sessionPreference: ['London', 'Other'],
      description:
        'Fade setups that exploit failed attempts to leave balance. After a failed breakout above or below value and a reclaim back inside, the agent trades back toward POC using LVNs on the reclaim leg and order-flow evidence that the breakout side cannot sustain.',
      steps: [
        'Define balance using prior day or session profile: identify POC, VAH and VAL.',
        'Confirm price has spent meaningful time rotating around POC, indicating balance.',
        'Detect breakout attempt beyond VAH or VAL that fails to sustain displacement.',
        'Recognize a second failure to remain outside value, confirming breakout failure.',
        'Define the reclaim leg back into balance and build a profile on that leg to identify LVNs.',
        'Wait for a pullback into a reclaim LVN and confirm that breakout-side aggression is failing or absorbed.',
        'Enter back toward POC with the stop just beyond the failed high or low and the confirming prints.',
        'Exit at POC under normal conditions without stretching to the other side of the range.'
      ],
      locationLogic:
        'Location is defined first by the value area (VAH, VAL, POC) of the balance, then by LVNs inside the reclaim leg after price has re-entered balance.',
      orderFlowLogic:
        'At the reclaim LVN, the breakout side should fail: aggression appears but cannot push price further, or dries up. The agent interprets this as confirmation that the attempt to leave balance has failed, and that rotation back to POC is likely.',
      riskManagementNotes: [
        'Stops sit just beyond the failed breakout high or low, with a small tick buffer.',
        'If price quickly returns outside value and holds beyond the LVN and failed extreme, the trade is invalid.',
        'Stop widening is prohibited; mean reversion trades are either right quickly or closed.'
      ],
      targetLogic:
        'The target is the balance POC. Extending to the opposite side of the range is viewed as stretching and is only considered in exceptional circumstances.'
    }
  ],
  examples: [
    {
      exampleId: 'trend_model_long_01',
      model: 'trend_continuation',
      session: 'NewYork',
      marketState: 'out_of_balance_uptrend',
      context:
        'Price broke out of prior balance and is strongly trending higher; the impulse leg from breakout to current high is clearly defined and trades above the old value area.',
      entryLogic:
        'Volume profile applied to the impulse leg reveals internal LVNs. Price pulls back into the first LVN. At that level, large buy prints and a buy-side footprint imbalance appear with CVD sloping upwards. A long is opened once the buy aggression is confirmed at the LVN.',
      riskManagement:
        'Stop is placed a few ticks below the aggressive buy prints and the local swing low. Risk per trade is between 0.25% and 0.5% of account. If CVD continues to accelerate upward after entry, the stop is advanced to break-even sooner.',
      exitLogic:
        'Planned target is the POC of the next balance area above. In the example, the position is reduced or closed when sellers show notable aggression against the move, illustrating that discretionary exit before POC is allowed when flow flips.'
    },
    {
      exampleId: 'mean_reversion_short_01',
      model: 'mean_reversion',
      session: 'London',
      marketState: 'balanced_with_failed_breakout_above',
      context:
        'Market is rotating around prior balance POC. Two separate upside attempts trade above VAH but both fail, with price re-entering the value area each time.',
      entryLogic:
        'After the second failed upside breakout, the reclaim leg back into balance is profiled. An LVN forms inside that reclaim leg. Price later pulls back up into this LVN. At that point buyer aggression is present initially but quickly fades without follow-through, confirming failure. A short is taken back toward POC.',
      riskManagement:
        'Stop is located just above the failed breakout high with an extra tick buffer. Risk per trade remains 0.25%–0.5% of the account. If price holds above the failed high and outside balance, the trade is considered invalid.',
      exitLogic:
        'Target is the balance POC. The position is exited at POC without attempting to stretch to the lower edge of the range.'
    }
  ],
  metaPros: [
    'Clear three-layer decision process: market state, location, and aggression must all align before trading.',
    'Adaptable to different days: trend continuation on out-of-balance days and mean reversion on rotational days.',
    'Tight, well-defined risk with stops anchored to concrete aggression zones and structure.',
    'High sample size over time because setups repeat across sessions and instruments.',
    'Built-in discipline via explicit no-trade conditions when alignment is missing.',
    'Return profiles that can reach 1:2.5 to 1:5 and occasionally higher on strong trend days.'
  ],
  metaCons: [
    'Win rate can be lumpy, with clusters of small stopouts in noisy conditions.',
    'Requires continuous monitoring of order flow, volume profile, and sessions, which is attention-intensive.',
    'Scaling position size increases psychological pressure even when rules remain identical.'
  ]
};

