import express, { Request, Response } from 'express';
import cors from 'cors';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const port = process.env.PORT || 3003;

app.use(cors());
app.use(express.json());

import menuRoutes from './routes/menu';

app.get('/', (req: Request, res: Response) => {
  res.send('Raw Juice Cafe Backend is running!');
});

app.use('/api/menu', menuRoutes);

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
