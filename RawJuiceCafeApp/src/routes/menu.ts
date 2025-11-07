
import express from 'express';
import { getMenu } from '../controllers/menuController';

const router = express.Router();

router.get('/', getMenu);

export default router;
